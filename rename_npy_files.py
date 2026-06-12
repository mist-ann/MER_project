import os
import pandas as pd

CSV_PATH = "/home/mist/Documents/datasets/addinfoMemo/songs_info_all.csv"
NPY_FOLDER = "/home/mist/Documents/datasets/mel_spec_processed"

df = pd.read_csv(CSV_PATH)

uuid_to_id = {
    row["file_name"].replace(".mp3", ""): str(row["song_id"])
    for _, row in df.iterrows()
}

renamed = 0
missing = 0

for file in os.listdir(NPY_FOLDER):
    if not file.endswith(".npy"):
        continue

    uuid = file.replace(".npy", "")

    if uuid in uuid_to_id:
        new_name = uuid_to_id[uuid] + ".npy"

        old_path = os.path.join(NPY_FOLDER, file)
        new_path = os.path.join(NPY_FOLDER, new_name)

        os.rename(old_path, new_path)
        renamed += 1
    else:
        print(f"⚠ No mapping for: {file}")
        missing += 1

print(f"\n✅ Renamed: {renamed}")
print(f"⚠ Missing mappings: {missing}")