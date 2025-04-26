## declare an array variable
declare -a exts=("wav" "mp3" "mp4" "ogg" "flac")

## now loop through the above array
for i in "${exts[@]}"
do
    python benchmark_pytorch.py --ext "$i"
done
