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

# STEP 3: perform data augmentation. Raw input data -> sentences txt file -> augmented sentences pkl file
# run each code piece in one machine. process data in parallel.


# # debug
# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
# data_type="squad"
# data_file_prefix="train"
# st_idx=0
# ed_idx=10000
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 -u QG_gpt2_generate_ind_acs_clue.py  \
#     --model_type gpt2 \
#     --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs/train_clue/ \
#     --filename "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ans.pkl" \
#     --data_type augmented_sents \
#     --output_file "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_clue.txt" \
#     --top_p 0.0 \
#     --save_freq 100 \
#     --debug \
#     --debug_num 100 &> experiments_4_QG_generate_gpt2_ind_acs_clue



# # squad data
# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
# data_type="squad"
# data_file_prefix="train"
# st_idx=0
# ed_idx=10000
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 -u QG_gpt2_generate_ind_acs_clue.py  \
#     --model_type gpt2 \
#     --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs/train_clue/ \
#     --filename "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ans.debug.pkl" \
#     --data_type augmented_sents \
#     --output_file "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ans.txt" \
#     --top_p 0.0 \
#     &>> experiments_4_QG_generate_gpt2_ind_acs_clue


# # wiki data
# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=0
# ed_idx=10000
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 -u QG_gpt2_generate_ind_acs_clue.py  \
#     --model_type gpt2 \
#     --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs/train_clue/ \
#     --filename "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ans.pkl" \
#     --data_type augmented_sents \
#     --output_file "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_clue.txt" \
#     --top_p 0.0 \
#     --save_freq 100 \
#     --debug \
#     --debug_num 500 &>> experiments_4_QG_generate_gpt2_ind_acs_clue


# # debug
# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1/"
# data_type="squad"
# data_file_prefix="dev"
# st_idx=0
# ed_idx=2000
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 -u QG_gpt2_generate_ind_acs_clue.py  \
#     --model_type gpt2 \
#     --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs__full_para/train_clue/ \
#     --filename "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ans.pkl" \
#     --data_type augmented_sents \
#     --output_file "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_clue.txt" \
#     --top_p 0.0 \
#     --save_freq 100 \
#     --debug \
#     --debug_num 500 &> experiments_4_QG_generate_gpt2_ind_acs_clue


input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1/"
output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1/"
data_type="squad"
data_file_prefix="train"
st_idx=0
ed_idx=2000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 -u QG_gpt2_generate_ind_acs_clue.py  \
    --model_type gpt2 \
    --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/file/QG/gpt2_ind_acs__full_para/train_clue/ \
    --filename "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_ans.pkl" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.ind_clue.txt" \
    --top_p 0.0 \
    --save_freq 100 \
    --debug \
    --debug_num 500 &> experiments_4_QG_generate_gpt2_ind_acs_clue