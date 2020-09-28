partition=${1}
config=${2}

job_name='topic-model'

srun --mpi=pmi2 -p ${partition} --gres=gpu:1 --job-name=${job_name} \
	python train_main.py -data_tag StackExchange_s150_t10 -only_train_ntm -ntm_warm_up_epochs ${config}
