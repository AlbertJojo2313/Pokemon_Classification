### pokemon_retriever.py
### author: Albert Jojo

"""
This script retrieves Pokémon sprite images from the PokeAPI based on a specified region.
It organizes the images in a directory structure and maintains a metadata.csv file with details about each image.
The main function is `get_pokemon_imgs(region)`, which takes a region name (e.g., "kanto", "johto", "hoenn", etc.) as input,
fetches the corresponding Pokémon names, and downloads their sprites while updating the metadata.csv file.
"""

import requests
import os
import csv
import io
from PIL import Image


# POKEMON_BASE = "https://pokemondb.net/pokedex/"


def _fetch_pokemon_by_region(region: str) -> list[tuple[int, str]]:
    region = region.lower()

    region_to_pokedex = {
        "kanto": ["kanto"],
        "johto": ["updated-johto"],
        "hoenn": ["hoenn"],
        "sinnoh": ["extended-sinnoh"],
        "unova": ["updated-unova"],
        "kalos": ["kalos-central", "kalos-coastal", "kalos-mountain"],
        "alola": ["updated-alola"],
        "galar": ["galar"],
        "paldea": ["paldea"],
    }

    if region not in region_to_pokedex:
        raise ValueError(f"Unknown region: {region}")

    pokedex_list = region_to_pokedex[region]

    pokemon_dict = {}

    for pokedex_name in pokedex_list:
        url = f"https://pokeapi.co/api/v2/pokedex/{pokedex_name}/"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        for entry in data["pokemon_entries"]:
            dex_id = entry["entry_number"]
            name = entry["pokemon_species"]["name"]

            pokemon_dict[dex_id] = name

    return sorted(pokemon_dict.items())


"""

Input: region (str) - The Pokémon region to retrieve images for (e.g., "kanto", "johto", "hoenn", etc.)
Output: Downloads Pokémon sprites for the specified region and writes metadata to a CSV file.

Description: This script fetches Pokémon names based on the specified region, 
retrieves their sprite images from the PokeAPI, saves them in a directory called "pokemon_images", 
and updates a "metadata.csv" file with details about each image.   
"""


def get_pokemon_imgs(region, base_dir: str = "pokemon_images"):
    pokemon_entries = _fetch_pokemon_by_region(region)

    sprites_to_retrieve = {
        "front_default": ("sprite", "front"),
        "back_default": ("sprite", "back"),
        "front_shiny": ("sprite", "shiny_front"),
        "back_shiny": ("sprite", "shiny_back"),
    }

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, base_dir)
    os.makedirs(data_path, exist_ok=True)

    # Metadata setup
    metadata_path = os.path.join(data_path, "metadata.csv")
    write_header = not os.path.exists(metadata_path)

    with open(metadata_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(
                [
                    "dex_id",
                    "pokemon_name",
                    "variant",
                    "source",
                    "image_path",
                ]
            )

        for dex_id, name in pokemon_entries:
            url = f"https://pokeapi.co/api/v2/pokemon/{dex_id}/"

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                # Use API ID explicitly (avoid overwriting loop variable)
                api_id = data["id"]

                pok_folder = os.path.join(data_path, f"{api_id:03d}_{name}")
                os.makedirs(pok_folder, exist_ok=True)

                # --- SPRITES ---
                for sprite_key, (source, file_prefix) in sprites_to_retrieve.items():
                    sprite_url = data["sprites"].get(sprite_key)
                    if not sprite_url:
                        continue

                    img_resp = requests.get(sprite_url, timeout=10)
                    img_resp.raise_for_status()

                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

                    # Preserve sprite sharpness
                    img = img.resize((224, 224), Image.NEAREST)

                    image_filename = f"{file_prefix}.png"
                    img_path = os.path.join(pok_folder, image_filename)
                    img.save(img_path)

                    relative_path = os.path.join(f"{api_id:03d}_{name}", image_filename)

                    writer.writerow(
                        [
                            api_id,
                            name,
                            file_prefix,
                            source,
                            relative_path,
                        ]
                    )

                    print(f"Saved sprite: {img_path}")

                # --- OFFICIAL ARTWORK ---
                official_artwork = (
                    data.get("sprites", {})
                    .get("other", {})
                    .get("official-artwork", {})
                    .get("front_default")
                )

                if official_artwork:
                    img_resp = requests.get(official_artwork, timeout=10)
                    img_resp.raise_for_status()

                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    img = img.resize((224, 224), Image.BILINEAR)

                    image_filename = "official_artwork.png"
                    img_path = os.path.join(pok_folder, image_filename)
                    img.save(img_path)

                    relative_path = os.path.join(f"{api_id:03d}_{name}", image_filename)

                    writer.writerow(
                        [
                            api_id,
                            name,
                            "official_art",
                            "official-artwork",
                            relative_path,
                        ]
                    )

                    print(f"Saved official artwork: {img_path}")

                # --- POKÉMON HOME ARTWORK ---
                home_art = (
                    data.get("sprites", {})
                    .get("other", {})
                    .get("home", {})
                    .get("front_default")
                )

                if home_art:
                    img_resp = requests.get(home_art, timeout=10)
                    img_resp.raise_for_status()

                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    img = img.resize((224, 224), Image.BILINEAR)

                    image_filename = "home_art.png"
                    img_path = os.path.join(pok_folder, image_filename)
                    img.save(img_path)

                    relative_path = os.path.join(f"{api_id:03d}_{name}", image_filename)

                    writer.writerow(
                        [
                            api_id,
                            name,
                            "home_art",
                            "pokemon_home",
                            relative_path,
                        ]
                    )

                    print(f"Saved Pokemon HOME artwork: {img_path}")

            except requests.RequestException as e:
                print(f"Failed to retrieve data for {name}: {e}")
                continue


def main():
    region = "kanto"

    print(f"Retrieving Pokémon images for region: {region}")
    get_pokemon_imgs(region)
    print(f"Completed retrieval for region: {region}\n")


if __name__ == "__main__":
    main()
