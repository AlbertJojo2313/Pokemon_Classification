import pandas as pd
import os
import shutil

TEST_IMAGES_PTH = 'test_images/images/'

def filter_dataset():
    df = pd.read_csv('test_images/pokemon.csv')

    # 1. Filter until Arceus
    arceus_idx = df[df['Name'].str.lower() == 'arceus'].index[0]
    df_filtered = df.iloc[:arceus_idx + 1].reset_index(drop=True)
    valid_pokemon = set(df_filtered['Name'].str.lower().str.strip())

    print(f"Pokémon kept (up to Arceus): {len(df_filtered)}")
    print(f"Last entry: {df_filtered['Name'].iloc[-1]}\n")

    images_folder = TEST_IMAGES_PTH
    all_images = os.listdir(images_folder)

    # 2. Build a lookup dict: pokemon_name -> filename
    # Try full stem first (handles nidoran-f, mr-mime, ho-oh etc.)
    # then fall back to base split (handles aegislash-blade -> aegislash)
    name_to_file = {}
    for filename in all_images:
        stem = os.path.splitext(filename)[0].lower().strip()

        # Full match first (e.g. nidoran-f, mr-mime, ho-oh)
        if stem in valid_pokemon:
            name_to_file[stem] = filename
        # Fallback: base name before '-' (e.g. aegislash-blade -> aegislash)
        else:
            base = stem.split('-')[0]
            if base in valid_pokemon and base not in name_to_file:
                name_to_file[base] = filename

    # 3. Build ordered image list matching CSV order
    ordered_images = []
    missing = []

    for pokemon_name in df_filtered['Name'].str.lower().str.strip():
        if pokemon_name in name_to_file:
            ordered_images.append(name_to_file[pokemon_name])
        else:
            missing.append(pokemon_name)

    print(f"✅ Matched & ordered: {len(ordered_images)}")
    print(f"⚠️  Still missing:    {len(missing)}")
    if missing:
        print(f"   Missing: {missing}")

    # 4. Delete images not in valid set
    removed = []
    for filename in all_images:
        if filename not in ordered_images:
            os.remove(os.path.join(images_folder, filename))
            removed.append(filename)

    print(f"❌ Removed: {len(removed)}")

    # 5. Rename files with zero-padded index to enforce CSV order
    print("\nReordering images to match CSV...")
    for idx, filename in enumerate(ordered_images):
        ext = os.path.splitext(filename)[1]
        new_name = f"{idx:03d}_{filename}"  # e.g. 000_bulbasaur.png
        os.rename(
            os.path.join(images_folder, filename),
            os.path.join(images_folder, new_name)
        )

    print("\nFirst 10 images in final order:")
    final_images = sorted(os.listdir(images_folder))
    for f in final_images[:10]:
        print(f"  {f}")
        
    # 6. Save filtered CSV with only Name column
    df_filtered = df_filtered.drop(columns=['Type1', 'Type2', 'Evolution'])
    df_filtered.to_csv('test_images/pokemon_filtered.csv', index=False)

    print(f"Saved filtered CSV: {len(df_filtered)} rows, columns: {df_filtered.columns.tolist()}")
    print("\n✅ Done! Dataset filtered and ordered.")

if __name__ == "__main__":
    filter_dataset()