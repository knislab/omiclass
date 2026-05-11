path=$(realpath ..)
data_path=$(realpath ../data)
cd $path || exit 1
export PYTHONPATH=$path:$PYTHONPATH
python api.py \
--override expression_file /app/data/merged_count_data.csv \
--override output_dir /app/results/Bash_results \
--override meta_path /app/data/merged_meta_with_external.csv