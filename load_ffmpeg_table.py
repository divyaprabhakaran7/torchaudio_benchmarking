import pandas as pd

# Load the benchmarking pickle file containing the results
benchmark_file_path = "/Users/Divya/torchaudio_benchmarking/results/benchmark_pytorch_wav.pickle"
benchmark_data = pd.read_pickle(benchmark_file_path)

# Assuming the structure of the benchmark data from your previous script is similar to this:
# 'total_file_size_KB', 'time' as columns (modify this part if necessary)
sizes_kb = benchmark_data['total_file_size_KB'].tolist()
times_ns = benchmark_data['time'].tolist()
libs = benchmark_data['lib'].tolist()

# Trim both lists to the smallest length
min_len = min(len(sizes_kb), len(times_ns))
sizes_kb = sizes_kb[:min_len]
times_ns = times_ns[:min_len]
libs = libs[:min_len]

# Assuming paired and sorted as before
paired = list(zip(sizes_kb, times_ns))
paired.sort(key=lambda x: x[1])  # sort by load time

# Create DataFrame
df = pd.DataFrame({
    'Library': libs,
    'File Size (KB)': sizes_kb,
    'Load Time (ns)': times_ns
})

# Save to CSV
df.to_csv('load_time_vs_size.csv', index=False)

print("CSV file 'load_time_vs_size.csv' saved successfully!")


