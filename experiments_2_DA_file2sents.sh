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

# # STEP 3: perform data augmentation. Raw input data -> sentences txt file -> augmented sentences pkl file

python3 -u DA_main.py \
        --da_task file2sentences \
        --da_input_type wiki10000 \
        --da_input_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/wiki10000.json \
        --da_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.txt \
        --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
        --not_processed_sample_probs_file \
        &> experiments_2_DA_wiki2sents
        
        

python3 -u DA_main.py \
        --da_task file2sentences \
        --da_input_type squad \
        --da_input_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/train-v2.0.json \
        --da_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.txt \
        --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.paragraphs.txt \
        &> experiments_2_DA_squad2sents
