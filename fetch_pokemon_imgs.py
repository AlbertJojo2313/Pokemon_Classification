### fetch_pokemon_data.py
### author: Albert Jojo

"""
Entry point for downloading all Pokémon image data.
Orchestrates pokemon_retrieval.py and pokemon_scraper.py for each region.

Usage:
    # Run all regions
    python fetch_pokemon_data.py

    # Run specific region(s)
    python fetch_pokemon_data.py kanto
    python fetch_pokemon_data.py kanto johto
"""

import argparse

from data.image_retrieval import get_pokemon_imgs
from data.image_scraper import scrape_extra_images


ALL_REGIONS = ["kanto", "johto", "hoenn", "sinnoh"]


def fetch_all(regions: list[str] = ALL_REGIONS):
    for region in regions:
        print(f"\n{'#' * 60}")
        print(f"  Region: {region.upper()}")
        print(f"{'#' * 60}")

        print("\n>>> Step 1: Fetching sprites and artwork from PokeAPI...")
        get_pokemon_imgs(region)

        print("\n>>> Step 2: Scraping additional artwork...")
        scrape_extra_images(region)

        print(f"\n✓ Finished {region}")

    print(f"\n{'#' * 60}")
    print(" All regions complete.")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Pokémon images for one or more regions."
    )
    parser.add_argument(
        "regions",
        nargs="*",
        choices=ALL_REGIONS,
        default=ALL_REGIONS,
        help="Regions to fetch. Defaults to all if not specified. "
        "e.g. python fetch_pokemon_data.py kanto johto",
    )
    args = parser.parse_args()
    fetch_all(args.regions)
