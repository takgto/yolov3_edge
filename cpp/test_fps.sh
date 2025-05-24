
# !/bin/bash
# usage: ./test_fps.sh <cpp_file_name_only> <xmodel_file>

# 0. make new directory (<unixtime>_test) where the cpp code and executable will be copied
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
echo "0. make new directory (../test/<unixtime>_test) where the cpp code and executable will be copied"
unixtime=$(date +%s)
mkdir -p "../test/${unixtime}_test"
touch "../test/${unixtime}_test/meta.txt"
# write the meta info to the file
echo "cpp_file: ${cpp_file}" >> "../test/${unixtime}_test/meta.txt"
echo "xmodel_file: ${xmodel_file}" >> "../test/${unixtime}_test/meta.txt"

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
cp ${cpp_dir} "../test/${unixtime}_test/" # copy the executable
cp ${cpp_file} "../test/${unixtime}_test/" # copy the cpp code

# 2.
echo "2. execute cpp code with the xmodel file: ${xmodel_file}"
for video_file in $(ls video/*.webm); do
    echo "video file: ${video_file}"
    vf=$(basename $video_file | cut -d'.' -f1)
    "./${cpp_dir}" $xmodel_file ./$video_file
    cp bench_tmp.csv ../test/${unixtime}_test/
    mv ../test/${unixtime}_test/bench_tmp.csv ../test/${unixtime}_test/${vf}_result.csv
    echo "result saved to ../test/${unixtime}_test/${vf}_result.csv"
    rm bench_tmp.csv
done

# 3.
echo "3. calculate the fps and execute time stats"
example_video="traffic1"
python3 "../test/stats.py" "../test/${unixtime}_test/"
python3 "../test/gantt_chart.py" "../test/${unixtime}_test/${example_video}_result.csv" --frames 10
python3 "../test/box_plot.py" "../test/${unixtime}_test/${example_video}_result.csv"

echo "Done. results in $test_dir"