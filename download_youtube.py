import pandas as pd
import subprocess
import os
from pathlib import Path
import sys
import time
from tqdm import tqdm

# Configuration
CSV_FILE = "/home/mist/Documents/datasets/MuSE/muse_v3.csv"
OUTPUT_DIR = "/home/mist/Documents/datasets/MuSE_spotify_previews"
AUDIO_FORMAT = "mp3"
NUM_SONGS = 500  # Download 500 songs
MAX_AUDIO_LENGTH = 120  # Maximum 2 minutes (shortend versions for testing)

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Read the CSV
df = pd.read_csv(CSV_FILE)

# Take first 500 songs for testing
df_test = df.head(NUM_SONGS).copy()


print("MUSIC EMOTION RECOGNITION - YouTube Download Pipeline (500 Songs)\n")
print(f"\nConfiguration:")
print(f"  CSV file: {CSV_FILE}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"Using first {len(df_test)} songs\n")


# Check if yt-dlp is available
try:
    result = subprocess.run(['yt-dlp', '--version'], capture_output=True, timeout=5, text=True)
    version = result.stdout.strip()
    print(f"✓ yt-dlp is installed: {version}\n")
except FileNotFoundError:
    print("✗ yt-dlp not found. Please install it:")
    print("  pip install --upgrade yt-dlp")
    sys.exit(1)

print("DOWNLOADING 500 SONGS")

successful = 0
failed = 0
skipped = 0
start_time = time.time()

# Progress bar
pbar = tqdm(total=len(df_test), desc="Overall Progress")

for idx, row in df_test.iterrows():
    track = row['track']
    artist = row['artist']
    valence = row['valence_tags']
    arousal = row['arousal_tags']
    
    # Create filename with emotion values for reference
    filename = f"{idx:03d}_{artist}_{track}".replace('/', '_').replace('\\', '_')[:120]
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.{AUDIO_FORMAT}")
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        skipped += 1
        pbar.update(1)
        pbar.set_description(f"Skipped: {skipped} | Downloaded: {successful} | Failed: {failed}")
        continue
    
    # Search query for YouTube
    search_query = f"{artist} {track} official"
    
    try:
        # yt-dlp command with minimal output
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio/best',
            '-x',
            '--audio-format', AUDIO_FORMAT,
            '--audio-quality', '192',
            '--max-filesize', '50M',
            '-o', os.path.join(OUTPUT_DIR, f"{filename}.%(ext)s"),
            f'ytsearch1:{search_query}',
            '--quiet',
            '--no-warnings',
            '--no-playlist'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(filepath):
            successful += 1
        else:
            failed += 1
    
    except subprocess.TimeoutExpired:
        failed += 1
    except Exception as e:
        failed += 1
    
    pbar.update(1)
    pbar.set_description(f"Skipped: {skipped} | Downloaded: {successful} | Failed: {failed}")

pbar.close()

# Final summary
elapsed_time = time.time() - start_time
elapsed_min = int(elapsed_time // 60)
elapsed_sec = int(elapsed_time % 60)

print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
print(f"✓ Successful: {successful}")
print(f"✗ Failed: {failed}")
print(f"⊘ Skipped: {skipped}")
print(f"─ Total processed: {successful + failed + skipped}")
print(f"\nTime elapsed: {elapsed_min}m {elapsed_sec}s")
print(f"Avg time per song: {elapsed_time/(successful + failed):.1f}s")
print(f"Success rate: {(successful/(successful+failed))*100:.1f}%" if (successful+failed) > 0 else "")
print(f"\nFiles location: {os.path.abspath(OUTPUT_DIR)}")

# List downloaded files
audio_files = list(Path(OUTPUT_DIR).glob(f"*.{AUDIO_FORMAT}"))
if len(audio_files) > 0:
    total_size = sum(f.stat().st_size for f in audio_files) / (1024 * 1024)
    print(f"\nTotal audio files: {len(audio_files)}")
    print(f"Total size: {total_size:.1f} MB")
    if len(audio_files) > 0:
        print(f"Avg file size: {total_size/len(audio_files):.1f} MB per file")
    
    print(f"\nSample files:")
    for f in sorted(audio_files)[:5]:
        print(f"  - {f.name}")

# Create a metadata file linking downloaded files to emotion values
print("\n" + "=" * 70)
print("CREATING METADATA")
print("=" * 70)

metadata = []
for idx, row in df_test.iterrows():
    track = row['track']
    artist = row['artist']
    filename = f"{idx:03d}_{artist}_{track}".replace('/', '_').replace('\\', '_')[:120]
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.{AUDIO_FORMAT}")
    
    if os.path.exists(filepath):
        metadata.append({
            'filename': f"{filename}.{AUDIO_FORMAT}",
            'artist': artist,
            'track': track,
            'valence': row['valence_tags'],
            'arousal': row['arousal_tags'],
            'dominance': row['dominance_tags'],
            'spotify_id': row['spotify_id'] if pd.notna(row['spotify_id']) else '',
        })

metadata_df = pd.DataFrame(metadata)
metadata_file = os.path.join(OUTPUT_DIR, 'metadata.csv')
metadata_df.to_csv(metadata_file, index=False)
print(f"\n✓ Metadata saved to: {metadata_file}")
print(f"✓ Total tracks with metadata: {len(metadata_df)}")

# Print emotion statistics
if len(metadata_df) > 0:
    print("\n" + "=" * 70)
    print("EMOTION LABEL STATISTICS")
    print("=" * 70)
    print(f"\nValence (positivity):")
    print(f"  Mean: {metadata_df['valence'].mean():.2f}")
    print(f"  Std: {metadata_df['valence'].std():.2f}")
    print(f"  Range: [{metadata_df['valence'].min():.2f}, {metadata_df['valence'].max():.2f}]")
    
    print(f"\nArousal (energy/intensity):")
    print(f"  Mean: {metadata_df['arousal'].mean():.2f}")
    print(f"  Std: {metadata_df['arousal'].std():.2f}")
    print(f"  Range: [{metadata_df['arousal'].min():.2f}, {metadata_df['arousal'].max():.2f}]")
    
    print(f"\nDominance (control/power):")
    print(f"  Mean: {metadata_df['dominance'].mean():.2f}")
    print(f"  Std: {metadata_df['dominance'].std():.2f}")
    print(f"  Range: [{metadata_df['dominance'].min():.2f}, {metadata_df['dominance'].max():.2f}]")
