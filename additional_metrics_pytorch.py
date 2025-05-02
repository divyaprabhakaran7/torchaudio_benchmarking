import matplotlib
matplotlib.use('Agg')
import torch.utils
import os
import os.path
import random
import time
import argparse
import librosa
import utils
import loaders
import torch
import glob
import psutil

def get_files(root_dir, extension):
    root_dir = os.path.expanduser(root_dir)
    pattern = os.path.join(root_dir, '**', f'*.{extension}')
    files = glob.glob(pattern, recursive=True)
    print(f"[get_files] Found {len(files)} '*.{extension}' files under {root_dir}")
    return files

class AudioFolder(torch.utils.data.Dataset):
    def __init__(self, root, extension='wav', lib="librosa"):
        self.root = os.path.expanduser(root)
        self.data = []
        self.audio_files = get_files(self.root, extension)
        print(f"[AudioFolder] Loader='{lib}' | Directory='{self.root}' | Files={len(self.audio_files)}")
        self.loader_function = getattr(loaders, lib)

    def __getitem__(self, index):
        fp = self.audio_files[index]
        try:
            audio = self.loader_function(fp)
        except Exception as e:
            print(f"[error] Loading '{fp}' with loader '{self.loader_function.__name__}': {e}")
            raise
        return torch.as_tensor(audio).view(1, 1, -1)

    def __len__(self):
        return len(self.audio_files)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ext', type=str, default="wav")
    args = parser.parse_args()

    repeat = 3
    columns = [
        'ext',
        'lib',
        'duration',
        'time',
        'mem_used_MB',
        'throughput_files_per_sec',
        'total_file_size_MB',
    ]

    store = utils.DF_writer(columns)

    libs = [
        'stempeg',
        'ar_ffmpeg',
    ]
    torchaudio_backends = ['torchaudio-ffmpeg', 'torchaudio-streamreader']
    libs.extend(torchaudio_backends)

    #if args.ext != "mp4":
       #libs.append('torchaudio-sox_io')
        #libs.append('torchaudio-soundfile')

    for lib in libs:
        print(f"\n===== Testing loader: {lib} =====")
        if "torchaudio" in lib:
            backend = lib.split("torchaudio-")[-1]
            if backend == 'streamreader':
                call_fun = 'load_torchaudio_streamreader'           
            else:
                import torchaudio
                torchaudio.set_audio_backend(backend)
                call_fun = "load_torchaudio"
        else:
            call_fun = 'load_' + lib

        for root, dirs, _ in sorted(os.walk('AUDIO')):
            print(f"[os.walk] Root: '{root}' | Subdirs: {dirs}")
            for audio_dir in dirs:
                print(f"[Directory] Processing '{audio_dir}' under '{root}'")
                try:
                    duration = int(audio_dir)
                except Exception as e:
                    print(f"[skip] Cannot parse duration from '{audio_dir}': {e}")
                    continue

                folder_path = os.path.join(root, audio_dir)
                dataset = AudioFolder(folder_path, extension=args.ext, lib=call_fun)
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False
                )
                print(f"[DataLoader] Created for duration={duration}s | Num_files={len(dataset)}")

                # Calculate total file size
                total_file_size = 0
                for fp in dataset.audio_files:
                    try:
                        total_file_size += os.path.getsize(fp)
                    except Exception as e:
                        print(f"[error] Getting size for {fp}: {e}")

                process = psutil.Process(os.getpid())
                start_mem = process.memory_info().rss
                start = time.time()

                for i in range(repeat):
                    for idx, fp in enumerate(dataset.audio_files):
                        try:
                            audio = dataset.loader_function(fp)
                            _ = torch.as_tensor(audio).view(1,1,-1).max()
                        except Exception as e:
                            print(f"[error] Iteration {i}, file {fp}: {e}")

                end = time.time()
                end_mem = process.memory_info().rss

                total_calls = len(dataset) * repeat
                avg_time = (end - start) / total_calls if total_calls > 0 else float('nan')
                mem_used = max(0.0, (end_mem - start_mem) / 1024)  # KB
                throughput = total_calls / (end - start) if (end - start) > 0 else float('nan')

                print(f"[Timing] lib={lib} | duration={duration}s | avg_time={avg_time:.6f}s per file | mem_used={mem_used:.2f}MB | throughput={throughput:.2f} files/sec | total_size={total_file_size/1024/1024:.2f}MB")

                store.append(
                    ext=args.ext,
                    lib=lib,
                    duration=duration,
                    time=avg_time,
                    mem_used_MB=mem_used,
                    throughput_files_per_sec=throughput,
                    total_file_size_KB=total_file_size / 1024,
                )

    os.makedirs("results", exist_ok=True)
    out_path = f"results/benchmark_pytorch_{args.ext}.pickle"
    store.df.to_pickle(out_path)
    print(f"Benchmark results saved to: {out_path}")
