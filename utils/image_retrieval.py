### pokemon_retriever.py
### author: Albert Jojo
import requests
import os
import csv

"""
This script retrieves Pokémon sprite images from the PokeAPI based on a specified region.
It organizes the images in a directory structure and maintains a metadata.csv file with details about each image.
The main function is `get_pokemon_imgs(region)`, which takes a region name (e.g., "kanto", "johto", "hoenn", etc.) as input,
fetches the corresponding Pokémon names, and downloads their sprites while updating the metadata.csv file.
"""


def _fetch_pokemon_by_region(region: str) -> list[str]:
    """Return a list of Pokémon names for a given region."""
    region = region.lower()

    region_to_pokedex = {
        "kanto": "kanto",  # urls
        "johto": "updated-johto",
        "hoenn": "hoenn",
        "sinnoh": "extended-sinnoh",
        "unova": "updated-unova",
        "kalos": ["kalos-central", "kalos-coastal", "kalos-mountain"],
        "alola": "updated-alola",
        "galar": "galar",
        "paldea": "paldea",
    }
    if region not in region_to_pokedex.keys():
        raise ValueError(f"Unknown region: {region}")

    url = f"https://pokeapi.co/api/v2/pokedex/{region_to_pokedex.get(region)}/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    pokemon_names = [
        entry["pokemon_species"]["name"] for entry in data["pokemon_entries"]
    ]

    return pokemon_names


"""

Input: region (str) - The Pokémon region to retrieve images for (e.g., "kanto", "johto", "hoenn", etc.)
Output: Downloads Pokémon sprites for the specified region and writes metadata to a CSV file.

Description: This script fetches Pokémon names based on the specified region, 
retrieves their sprite images from the PokeAPI, saves them in a directory called "pokemon_images", 
and updates a "metadata.csv" file with details about each image.   
"""


def get_pokemon_imgs(region, base_dir: str = "pokemon_images"):
    pokemon_names = _fetch_pokemon_by_region(region)

    sprites_to_retrieve = {
        "front_default": ("front", "front"),
        "back_default": ("back", "back"),
        "front_shiny": ("shiny", "shiny_front"),
        "back_shiny": ("shiny", "shiny_back"),
    }
    # Get the directory for the scripts file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, base_dir)

    # Create the output directory
    os.makedirs(data_path, exist_ok=True)

    # Create/append to the metadata.csv file
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
                    "image_path",
                ]
            )

        for name in pokemon_names:
            url = f"https://pokeapi.co/api/v2/pokemon/{name}/"
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                dex_id = data["id"]

                # Create individual Pokémon folder:pokemon_images/001_bulbasaur
                pok_folder = os.path.join(data_path, f"{dex_id:03d}_{name}")
                os.makedirs(pok_folder, exist_ok=True)

                for sprite_key, (_, file_prefix) in sprites_to_retrieve.items():
                    sprite_url = data["sprites"][sprite_key]  # Get the sprite URL
                    if sprite_url is None:
                        continue

                    image_response = requests.get(sprite_url)
                    image_response.raise_for_status()

                    # Save as: pokemon_images/001_bulbasaur/front.png
                    image_filename = f"{file_prefix}.png"
                    img_path = os.path.join(pok_folder, image_filename)

                    with open(img_path, "wb") as img_file:
                        img_file.write(image_response.content)

                    # Store the relative path in metadata.csv
                    relative_path = os.path.join(f"{dex_id:03d}_{name}", image_filename)
                    writer.writerow(
                        [
                            dex_id,
                            name,
                            file_prefix,
                            relative_path,
                        ]
                    )
                    print(f"Saved {img_path} and updated metadata")
            except requests.HTTPError as e:
                print(f"Failed to retrieve data for {name}: {e}")
                continue


def main():
    get_pokemon_imgs("hoenn")


if __name__ == "__main__":
    main()
