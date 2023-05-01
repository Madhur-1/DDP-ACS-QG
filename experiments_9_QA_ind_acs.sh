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

output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
data_file_prefix="train"
st_idx=0
ed_idx=10000
python3 QA_generate.py \
        --model_name_or_path "/scratch/scratch8/madhurjindal/ACS-QG-Scratch/models/distilbert-base-cased-distilled-squad" \
        --input_file  "${output_path}${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ques.txt" \
        --output_file "${output_path}${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ques.qa.txt" \
        &> experiments_9_QA_ind_acs

output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=0
ed_idx=10000
python3 QA_generate.py \
        --model_name_or_path "/scratch/scratch8/madhurjindal/ACS-QG-Scratch/models/distilbert-base-cased-distilled-squad" \
        --input_file  "${output_path}${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ques.txt" \
        --output_file "${output_path}${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ques.qa.txt" \
        &>> experiments_9_QA_ind_acs
