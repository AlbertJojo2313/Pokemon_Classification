### pokemon_scraper.py
### author: Albert Jojo

"""
Scrapes additional artwork for Pokémon (Gen 1–4) from multiple sources.
Sprites are fully handled by pokemon_retrieval.py — this script adds artwork only.

Intentionally excluded (already covered by pokemon_retrieval.py):
    artwork_sugimori   — same as official_artwork.png from PokeAPI
    dream_world        — requires cairo system library

New files added per Pokémon:
    From PokemonDB (direct CDN URLs):
        artwork_gen1_us.jpg          — early Red/Blue US art (Kanto only)
        artwork_gen1_jp.jpg          — early Red/Green JP art (Kanto only)
        artwork_global_link.png      — Global Link vector/action pose

    From HybridShivam GitHub (high-res Sugimori from Bulbapedia):
        artwork_highres.png          — highest quality official Sugimori art
"""

import os
import csv
import time
import requests

from utils.image_retrieval import _fetch_pokemon_by_region


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (personal research project)"})

# PokemonDB artwork — direct CDN, no scraping needed
# artwork_sugimori intentionally excluded — same as official_artwork.png from PokeAPI
# gen1 early artworks only exist for Kanto — 404s skipped silently for other regions
POKEMONDB_ARTWORKS = [
    (
        "https://img.pokemondb.net/artwork/original/{name}-gen1.jpg",
        "artwork_gen1_us.jpg",
        "sugimori_gen1_us",
    ),
    (
        "https://img.pokemondb.net/artwork/original/{name}-gen1-jp.jpg",
        "artwork_gen1_jp.jpg",
        "sugimori_gen1_jp",
    ),
    (
        "https://img.pokemondb.net/artwork/vector/{name}.png",
        "artwork_global_link.png",
        "global_link",
    ),
]

# HybridShivam GitHub — high-res Sugimori artwork sourced from Bulbapedia
HYBRIDSHIVAM_BASE = "https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/images/{dex_id:03d}.png"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _download(url: str, dest_path: str, delay: float = 0.5) -> bool:
    if os.path.exists(dest_path):
        print(f"  skip (exists) → {os.path.basename(dest_path)}")
        return True
    try:
        r = SESSION.get(url, timeout=10, allow_redirects=True)
        time.sleep(delay)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            with open(dest_path, "wb") as f:
                f.write(r.content)
            print(f"  ✓ {os.path.basename(dest_path)}")
            return True
        print(f"  ✗ {os.path.basename(dest_path)} ({r.status_code})")
        return False
    except requests.RequestException as e:
        print(f"  ✗ {os.path.basename(dest_path)} (error: {e})")
        return False


def _write_metadata(writer, dex_id, name, variant, source, rel_path):
    writer.writerow([dex_id, name, variant, source, rel_path])


# ── 1. PokemonDB artwork ──────────────────────────────────────────────────────


def _scrape_pokemondb_artwork(folder_path, dex_id, name, writer, delay=0.6):
    print(f"\n  [pokemondb artwork]")
    for url_template, filename, variant_label in POKEMONDB_ARTWORKS:
        url = url_template.format(name=name)
        dest = os.path.join(folder_path, filename)
        ok = _download(url, dest, delay)
        if ok:
            rel_path = os.path.join(f"{dex_id:03d}_{name}", filename)
            _write_metadata(
                writer, dex_id, name, variant_label, "pokemondb_artwork", rel_path
            )


# ── 2. HybridShivam GitHub high-res artwork ──────────────────────────────────


def _scrape_hybridshivam_artwork(folder_path, dex_id, name, writer, delay=0.6):
    """
    Downloads high-res official Sugimori artwork from the HybridShivam/Pokemon
    GitHub repository, which sources images from Bulbapedia.
    """
    print(f"\n  [hybridshivam high-res artwork]")
    url = HYBRIDSHIVAM_BASE.format(dex_id=dex_id)
    dest = os.path.join(folder_path, "artwork_highres.png")
    ok = _download(url, dest, delay)
    if ok:
        rel_path = os.path.join(f"{dex_id:03d}_{name}", "artwork_highres.png")
        _write_metadata(
            writer, dex_id, name, "sugimori_highres", "hybridshivam_github", rel_path
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def scrape_extra_images(
    region: str, base_dir: str = "pokemon_images", delay: float = 0.6
):
    """
    Scrapes artwork for all Pokémon in the given region.
    Appends new rows to the existing metadata.csv.

    Input:  region (str) — "kanto", "johto", "hoenn", or "sinnoh"
    Output: Artwork saved into existing flat Pokémon folders, metadata.csv updated
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

            folder_path = os.path.join(data_path, f"{dex_id:03d}_{name}")
            os.makedirs(folder_path, exist_ok=True)

            _scrape_pokemondb_artwork(folder_path, dex_id, name, writer, delay)
            _scrape_hybridshivam_artwork(folder_path, dex_id, name, writer, delay)
