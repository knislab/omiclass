path=$(realpath ..)
data_path=$(realpath ./data)
cd $path || exit 1
export PYTHONPATH=$path:$PYTHONPATH
python api.py \
--override expression_file $data_path/raw_counts.csv \
--override output_dir ./examples/results \
--override meta_path $data_path/patients_meta.csv \
--override class_col class \
--override batch_column batch \
--override de_covariates age,sex \
--override n_jobs 5 \
--override gene_lengths_path $data_path/gene_lengths.csv