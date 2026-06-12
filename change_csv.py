import pandas as pd
import numpy as np

valence = pd.read_csv('/home/mist/Documents/datasets/addinfoMemo/valence_all_average.csv', index_col='song_id')
arousal = pd.read_csv('/home/mist/Documents/datasets/addinfoMemo/arousal_all_average.csv', index_col='song_id')
valence_std = pd.read_csv('/home/mist/Documents/datasets/addinfoMemo/valence_all_std.csv', index_col='song_id')
arousal_std = pd.read_csv('/home/mist/Documents/datasets/addinfoMemo/arousal_all_std.csv', index_col='song_id')

data = []

for song_id in valence.index:
    for i, col in enumerate(valence.columns):
        # Parsuj kolumnę: "sample_1000ms" → 1.0 sekund
        time_ms = int(col.replace('sample_', '').replace('ms', ''))
        time_sec = time_ms / 1000.0
        
        row = {
            'song_id': song_id,
            'time': time_sec,
            'valence_mean': valence.loc[song_id, col],
            'arousal_mean': arousal.loc[song_id, col],
            'valence_std': valence_std.loc[song_id, col],
            'arousal_std': arousal_std.loc[song_id, col]
        }
        data.append(row)

df = pd.DataFrame(data)
df.to_csv('sp2/labels.csv', index=False)

print(f"✓ Converted to long format: {len(df)} rows")
print(df.head(10))