path=$(realpath ..)
data_path=$(realpath ./data)
results_path=$(realpath ./results)
cd $path || exit 1
export PYTHONPATH=$path:$PYTHONPATH
python api.py \
--override expression_file $data_path/raw_counts_with_external.csv \
--override output_dir $results_path/Bash_results \
--override meta_path $data_path/patients_meta_with_external.csv \
--override raw_count True