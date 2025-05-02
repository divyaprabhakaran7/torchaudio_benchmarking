import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for package in ['pytorch']:
    dfs = []
    #for ext in ["wav", "mp3", "mp4", "ogg", "flac"]:
    for ext in ["wav"]:
        try:
            dfs.append(
                pd.read_pickle("results/benchmark_%s_%s.pickle" % (package, ext))
            )
        except FileNotFoundError:
            continue

    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("whitegrid")

    ordered_exts = df.time.groupby(df.ext).mean().sort_values().index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    metrics = [
        ("time", "Load Time per File (s, log scale)", True),
        ("mem_used_MB", "Memory Usage (KB)", False),
        ("throughput_files_per_sec", "Throughput (files/sec)", False),
    ]

    for ax, (metric, title, log_scale) in zip(axes, metrics):
        sns.barplot(
            x=metric,
            y="ext",
            hue="lib",
            order=ordered_exts,
            data=df,
            ax=ax
        )
        ax.set_title(title)
        if log_scale:
            ax.set_xscale("log")
        ax.legend(loc='best')
        ax.set_xlabel(metric)
        ax.set_ylabel("Extension")

    plt.tight_layout()
    plt.savefig("results/benchmark_%s.png" % package)
    plt.close()

