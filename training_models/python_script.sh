experiment_index=$1
echo $experiment_index
if [ "$experiment_index" = "0" ]; then
    eval "$(conda shell.bash hook)"
    conda activate diff_rendering_ml
    cd uniform_data_experiments
    python finding_failures.py
    # python data_size_means.py
    # python epochs_experiment.py
    #python data_dimension.py
fi
