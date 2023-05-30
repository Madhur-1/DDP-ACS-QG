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

# squad data

# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
# data_type="squad"
# data_file_prefix="train"
# st_idx=0
# ed_idx=10000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}-v2.0.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
# data_type="squad"
# data_file_prefix="train"
# st_idx=50000
# ed_idx=92210
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}-v2.0.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 

# # SQuAD1.1 Data
# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1/"
# data_type="squad"
# data_file_prefix="dev"
# st_idx=0
# ed_idx=2000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}-v1.1.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 


# SQuAD1.1 Data
input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD1.1/"
output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD1.1/"
data_type="squad"
data_file_prefix="train"
st_idx=0
ed_idx=2000
python3 -u DA_main.py \
        --da_task sentences2augmented_sentences \
        --da_input_type $data_type \
        --da_input_file "$input_path${data_file_prefix}-v1.1.json" \
        --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
        --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
        --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
        --da_start_index $st_idx \
        --da_end_index $ed_idx \
        &>> experiments_3_DA_sents2augsents 

# # wiki data

# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=0
# ed_idx=10000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=50000
# ed_idx=100000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=100000
# ed_idx=150000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=150000
# ed_idx=200000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=200000
# ed_idx=250000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=250000
# ed_idx=300000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=300000
# ed_idx=350000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=350000
# ed_idx=400000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=400000
# ed_idx=450000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=450000
# ed_idx=500000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=500000
# ed_idx=550000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=550000
# ed_idx=600000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=600000
# ed_idx=650000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=650000
# ed_idx=700000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=700000
# ed_idx=750000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=750000
# ed_idx=800000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=800000
# ed_idx=850000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=850000
# ed_idx=900000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 





# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=900000
# ed_idx=9500000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=950000
# ed_idx=1000000
# python3 -u DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type $data_type \
#         --da_input_file "$input_path${data_file_prefix}.json" \
#         --da_sentences_file "$output_path${data_file_prefix}.sentences.txt" \
#         --da_paragraphs_file "$output_path${data_file_prefix}.paragraphs.txt" \
#         --da_augmented_sentences_file "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
#         --da_start_index $st_idx \
#         --da_end_index $ed_idx \
#         &>> experiments_3_DA_sents2augsents 
