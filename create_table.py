import pandas as pd

dfs = []

# Iterate over different file extensions
for ext in ["wav", "mp3", "mp4", "ogg", "flac"]:
    try:
        # Read each pickle file and append to the list
        dfs.append(pd.read_pickle(f"results/benchmark_pytorch_{ext}.pickle"))
    except FileNotFoundError:
        continue

df = pd.concat(dfs, ignore_index=True)

grouped_df = df.groupby(['ext', 'lib']).mean().reset_index()

sorted_df = grouped_df.sort_values(by='time')

sorted_df.to_csv("results/benchmark_pytorch_sorted.csv", index=False)

