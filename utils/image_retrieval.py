### pokemon_retriever.py
### author: Albert Jojo

"""
Retrieves Pokémon images from PokeAPI for regions up to Sinnoh (Gen 1–4).

Images retrieved per Pokémon:
    Official artwork:
        official_artwork.png    — Ken Sugimori / modern official art
        home_art.png            — Pokémon HOME render

    Gen 6–7 sprites (high quality, consistent style, suitable for classification):
        gen6_xy_front.png, gen6_xy_shiny_front.png
        gen6_oras_front.png, gen6_oras_shiny_front.png
        gen7_usum_front.png, gen7_usum_shiny_front.png

The rest gens avoided due to low quality.
"""

import requests
import os
import csv
import io
from PIL import Image


GEN_SPRITE_MAP = [
    # Gen 6 — X/Y
    ("generation-vi", "x-y", "front_default", "gen6_xy_front.png", "BILINEAR"),
    ("generation-vi", "x-y", "front_shiny", "gen6_xy_shiny_front.png", "BILINEAR"),
    # Gen 6 — OmegaRuby/AlphaSapphire
    (
        "generation-vi",
        "omegaruby-alphasapphire",
        "front_default",
        "gen6_oras_front.png",
        "BILINEAR",
    ),
    (
        "generation-vi",
        "omegaruby-alphasapphire",
        "front_shiny",
        "gen6_oras_shiny_front.png",
        "BILINEAR",
    ),
    # Gen 7 — Ultra Sun/Ultra Moon
    (
        "generation-vii",
        "ultra-sun-ultra-moon",
        "front_default",
        "gen7_usum_front.png",
        "BILINEAR",
    ),
    (
        "generation-vii",
        "ultra-sun-ultra-moon",
        "front_shiny",
        "gen7_usum_shiny_front.png",
        "BILINEAR",
    ),
]


def _fetch_pokemon_by_region(region: str) -> list[tuple[int, str]]:
    region = region.lower()

    region_pokedex = {
        "kanto": (1, 151),
        "johto": (152, 251),
        "hoenn": (252, 386),
        "sinnoh": (387, 493),
    }

    if region not in region_pokedex:
        raise ValueError(f"Unknown region: {region}")

    start_num, end_num = region_pokedex[region]
    pokemon_list = []

    for dex_id in range(start_num, end_num + 1):
        try:
            response = requests.get(
                f"https://pokeapi.co/api/v2/pokemon/{dex_id}", timeout=10
            )
            response.raise_for_status()
            data = response.json()
            pokemon_list.append((dex_id, data["name"]))
        except requests.RequestException as e:
            print(f"  Failed to retrieve data for dex ID {dex_id}: {e}")

    return pokemon_list


def _download_and_save(
    img_url: str,
    img_path: str,
    resize: tuple = (224, 224),
    resample: str = "BILINEAR",
) -> bool:
    """Download an image from URL, resize, and save. Returns True on success."""
    if os.path.exists(img_path):
        print(f"  skip (exists) → {os.path.basename(img_path)}")
        return True
    try:
        resp = requests.get(img_url, timeout=10)
        resp.raise_for_status()
        resample_filter = Image.NEAREST if resample == "NEAREST" else Image.BILINEAR
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA").convert("RGB")
        img = img.resize(resize, resample_filter)
        img.save(img_path)
        print(f"  ✓ {os.path.basename(img_path)}")
        return True
    except Exception as e:
        print(f"  ✗ {os.path.basename(img_path)} ({e})")
        return False


def get_pokemon_imgs(region: str, base_dir: str = "pokemon_images"):
    """
    Downloads all Pokémon images for the given region into flat per-Pokémon folders.
    Appends rows to metadata.csv.

    Input:  region (str) — "kanto", "johto", "hoenn", or "sinnoh"
    Output: Images saved to {base_dir}/{dex_id:03d}_{name}/, metadata.csv updated
    """
    pokemon_entries = _fetch_pokemon_by_region(region)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # utils/ → project_root/
    data_path = os.path.join(project_root, base_dir)
    os.makedirs(data_path, exist_ok=True)

    metadata_path = os.path.join(data_path, "metadata.csv")
    write_header = not os.path.exists(metadata_path)

    with open(metadata_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(
                ["dex_id", "pokemon_name", "variant", "source", "image_path"]
            )

        for dex_id, name in pokemon_entries:
            print(f"\n{'=' * 50}")
            print(f"  {dex_id:03d} {name}")
            print(f"{'=' * 50}")

            try:
                data = requests.get(
                    f"https://pokeapi.co/api/v2/pokemon/{dex_id}/", timeout=10
                ).json()
            except requests.RequestException as e:
                print(f"  Failed to fetch data: {e}")
                continue

            api_id = data["id"]
            pok_folder = os.path.join(data_path, f"{api_id:03d}_{name}")
            os.makedirs(pok_folder, exist_ok=True)

            def save(url, filename, variant, source, resample="BILINEAR"):
                if not url:
                    return
                img_path = os.path.join(pok_folder, filename)
                ok = _download_and_save(url, img_path, resample=resample)
                if ok:
                    writer.writerow(
                        [
                            api_id,
                            name,
                            variant,
                            source,
                            os.path.join(f"{api_id:03d}_{name}", filename),
                        ]
                    )

            sprites = data["sprites"]
            other = sprites.get("other", {})
            versions = sprites.get("versions", {})

            # ── Official artwork ──────────────────────────────────────────
            print("\n  [artwork]")
            save(
                other.get("official-artwork", {}).get("front_default"),
                "official_artwork.png",
                "official_art",
                "official-artwork",
            )
            save(
                other.get("home", {}).get("front_default"),
                "home_art.png",
                "home_art",
                "pokemon_home",
            )

            # ── Gen 6–7 sprites (high quality, 3D style) ─────────────────
            print("\n  [gen sprites]")
            for gen_key, ver_key, sprite_key, filename, resample in GEN_SPRITE_MAP:
                url = versions.get(gen_key, {}).get(ver_key, {}).get(sprite_key)
                save(
                    url,
                    filename,
                    filename.replace(".png", ""),
                    "pokeapi_versions",
                    resample,
                )
