
# !/bin/bash
# usage: ./benchmark_fps.sh <cpp_file_name_only> <xmodel_file>

# 0. make new directory (<unixtime>_benchmark) where the cpp code and executable will be copied
# 1. compile the cpp code
#       copy the cpp code and executable to the directory
# 2. execute and output the result to a directory where the original cpp code is
# 3. calculate the fps and execute time stats

set -e  # エラー時に即終了

cpp_dir=$(basename $PWD)
cpp_file=$1
cpp_file_noext=$(echo $cpp_file | cut -d'.' -f1)
xmodel_file=$2
skip_compile=${3:-0}  # skip_compile is optional, default is 0
# make sure these files exist
if [ ! -f ${cpp_file} ]; then
    echo "cpp file not found: ${cpp_file}"
    exit 1
fi
if [ ! -f ${xmodel_file} ]; then
    echo "xmodel file not found: ${xmodel_file}"
    exit 1
fi

# 0.
echo "0. make new directory (../benchmark/<cpp_file_noext>_benchmark_<unixtime>) where the cpp code and executable will be copied"
unixtime=$(date ""+%m%d_%H%M%S"")
benchmark_dir="../benchmark/${cpp_file_noext}_benchmark_${unixtime}"
mkdir -p "${benchmark_dir}"
touch "${benchmark_dir}/meta.txt"

echo "cpp_file: ${cpp_file}" >> "${benchmark_dir}/meta.txt"
echo "xmodel_file: ${xmodel_file}" >> "${benchmark_dir}/meta.txt"

# 1.
echo "1. compile the cpp code"
if [ $skip_compile -eq 0 ]; then
    echo "compiling ${cpp_file}..."
    if [ ! -f ./make.sh ]; then
        echo "make.sh not found in the current directory. Please run this script from the directory containing make.sh."
        exit 1
    fi
    ./make.sh ${cpp_file}
else
    echo "skipping compilation as per user request."
fi
cp ${cpp_dir} "${benchmark_dir}/" # copy the executable
cp ${cpp_file} "${benchmark_dir}/" # copy the cpp code

# 2.
echo "2. execute cpp code with the xmodel file: ${xmodel_file}"
for video_file in $(ls video/*.webm); do
    echo "video file: ${video_file}"
    vf=$(basename $video_file | cut -d'.' -f1)
    "./${cpp_dir}" $xmodel_file ./$video_file
    cp bench_tmp.csv ${benchmark_dir}/
    mv ${benchmark_dir}/bench_tmp.csv ${benchmark_dir}/${vf}_result.csv
    echo "result saved to ${benchmark_dir}/${vf}_result.csv"
    rm bench_tmp.csv
done

# 3.
echo "3. calculate the fps and execute time stats"
example_video="traffic1"
python3 "../benchmark/stats.py" "${benchmark_dir}/"
python3 "../benchmark/gantt_chart.py" "${benchmark_dir}/${example_video}_result.csv" --frames 30
python3 "../benchmark/box_plot.py" "${benchmark_dir}/${example_video}_result.csv"

echo "Done. results in $benchmark_dir"