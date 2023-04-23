# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tools/cuda/lib64
# Tools config for CUDA, Anaconda installed in the common /tools directory
export TRANSFORMERS_CACHE=/scratch/scratch8/madhurjindal/transformers_cache

source /tools/config.sh

source activate /scratch/scratch8/madhurjindal/acs-qg-env

cd /storage/home/madhurjindal/ACS-QG


# python3 -u QG_gpt2_train_ind_acs.py \
#     --eval_before_start \
#     --n_epochs 4 \
#     --model_name_or_path gpt2 \
#     --output_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs/train_ans \
#     --train_dataset_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1-Zhou/train.txt \
#     --dev_dataset_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1-Zhou/dev.txt \
#     --train_dataset_cache /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1-Zhou/train-gpt2.pt \
#     --dev_dataset_cache /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1-Zhou/dev-gpt2.pt \
#     --local_rank -1 \
#     --train_batch_size 4 \
#     --train_target ans \
#     &> experiments_1_QG_gpt2_ind_acs_ans


# python3 -u QG_gpt2_train_ind_acs.py \
#     --eval_before_start \
#     --n_epochs 4 \
#     --model_name_or_path gpt2 \
#     --output_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs/train_clue \
#     --train_dataset_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1-Zhou/train.txt \
#     --dev_dataset_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1-Zhou/dev.txt \
#     --train_dataset_cache /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1-Zhou/train-gpt2.pt \
#     --dev_dataset_cache /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1-Zhou/dev-gpt2.pt \
#     --local_rank -1 \
#     --train_batch_size 4 \
#     --train_target clue \
#     &> experiments_1_QG_gpt2_ind_acs_clue

python3 -u QG_gpt2_train_ind_acs.py \
    --eval_before_start \
    --n_epochs 4 \
    --model_name_or_path gpt2 \
    --output_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs/train_style \
    --train_dataset_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1-Zhou/train.txt \
    --dev_dataset_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1-Zhou/dev.txt \
    --train_dataset_cache /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1-Zhou/train-gpt2.pt \
    --dev_dataset_cache /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1-Zhou/dev-gpt2.pt \
    --local_rank -1 \
    --train_batch_size 4 \
    --train_target style \
    &> experiments_1_QG_gpt2_ind_acs_style