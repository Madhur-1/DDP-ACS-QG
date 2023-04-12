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

# This script we should repeat it for different parts of data
# For example:
# 1. replace 'da_start_index 0' with 'da_start_index 10000'
# 2. replace 'da_end_index 10000' with 'da_end_index 20000'
# 3. replace '0_10000' with '10000_20000'
# Similarly, 20000~30000, 30000~40000, 40000~50000, .....


# # STEP 3: perform data augmentation. Raw input data -> sentences txt file -> augmented sentences pkl file

# python3 DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type wiki10000 \
#         --da_input_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/wiki10000.json \
#         --da_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.txt \
#         --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
#         --da_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.pkl \
#         --da_start_index 0 \
#         --da_end_index 10000 &> experiments_3_repeat_da_de

# python3 DA_main.py \
#         --da_task sentences2augmented_sentences \
#         --da_input_type squad \
#         --da_input_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/train-v2.0.json \
#         --da_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.txt \
#         --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.paragraphs.txt \
#         --da_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.pkl \
#         --da_start_index 0 \
#         --da_end_index 10000 \
#         &>> experiments_3_repeat_da_de

# # # STEP 4: use trained FQG model to generate new QG data using augmented sentences
# # prepro: it doesn't need GPU
# python3 QG_augment_main.py \
#         --not_processed_data  \
#         --batch_size 8 \
#         --epochs 10 \
#         --copy_type hard-oov \
#         --copy_loss_type 1 \
#         --use_style_info \
#         --use_clue_info \
#         -beam_size 20 \
#         --use_refine_copy_tgt_src \
#         --da_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.pkl \
#         --qg_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.pkl \
#         --qg_result_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.output.txt \
#         --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.paragraphs.txt \
#         --qa_data_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.txt \
#         --mode prepro \
#         &>> experiments_3_repeat_da_de


# python3 QG_augment_main.py \
#         --not_processed_data  \
#         --batch_size 8 \
#         --epochs 10 \
#         --copy_type hard-oov \
#         --copy_loss_type 1 \
#         --use_style_info \
#         --use_clue_info \
#         -beam_size 20 \
#         --use_refine_copy_tgt_src \
#         --da_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.pkl \
#         --qg_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.pkl \
#         --qg_result_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.output.txt \
#         --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
#         --qa_data_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.txt \
#         --mode prepro \
#         &>> experiments_3_repeat_da_de

# # generate: needs GPU
# python3 QG_augment_main.py \
#         --not_processed_data  \
#         --batch_size 8 \
#         --epochs 10 \
#         --copy_type hard-oov \
#         --copy_loss_type 1 \
#         --use_style_info \
#         --use_clue_info \
#         -beam_size 20 \
#         --use_refine_copy_tgt_src \
#         --da_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.pkl \
#         --qg_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.pkl \
#         --qg_result_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.output.txt \
#         --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.paragraphs.txt \
#         --qa_data_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.txt \
#         &>> experiments_3_repeat_da_de

# python3 QG_augment_main.py \
#         --not_processed_data  \
#         --batch_size 8 \
#         --epochs 10 \
#         --copy_type hard-oov \
#         --copy_loss_type 1 \
#         --use_style_info \
#         --use_clue_info \
#         -beam_size 20 \
#         --use_refine_copy_tgt_src \
#         --da_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.pkl \
#         --qg_augmented_sentences_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.pkl \
#         --qg_result_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.output.txt \
#         --da_paragraphs_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
#         --qa_data_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.txt \
#         &>> experiments_3_repeat_da_de

# sort /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.txt | uniq  > /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.uniq.txt
# sort /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.txt | uniq > /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.uniq.txt

# # STEP 5: use trained entailment model to append entailment score column
# python3 run_glue.py \
#         --model_type xlnet \
#         --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/models/xlnet-base-cased/ \
#         --task_name MRPC \
#         --do_test \
#         --do_lower_case \
#         --data_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/glue_data/squad-rte/MRPC/ \
#         --max_seq_length 128 \
#         --per_gpu_eval_batch_size=8   \
#         --per_gpu_train_batch_size=8   \
#         --learning_rate 2e-5 \
#         --num_train_epochs 1.0 \
#         --output_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/et_outdir/xlnet-base-cased/ \
#         --overwrite_output_dir \
#         --context_question_answer_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.uniq.txt \
#         --context_question_answer_columns 3 2 4 \
#         --context_question_answer_score_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.entail.txt \
#         &>> experiments_3_repeat_da_de

# python3 run_glue.py \
#         --model_type xlnet \
#         --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/models/xlnet-base-cased/ \
#         --task_name MRPC \
#         --do_test \
#         --do_lower_case \
#         --data_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/glue_data/squad-rte/MRPC/ \
#         --max_seq_length 128 \
#         --per_gpu_eval_batch_size=8   \
#         --per_gpu_train_batch_size=8   \
#         --learning_rate 2e-5 \
#         --num_train_epochs 1.0 \
#         --output_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/et_outdir/xlnet-base-cased/ \
#         --overwrite_output_dir \
#         --context_question_answer_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.uniq.txt \
#         --context_question_answer_columns 3 2 4 \
#         --context_question_answer_score_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.entail.txt \
#         &>> experiments_3_repeat_da_de

# STEP 6: perform data evaluation to filter low-quality data samples and tag data samples with quality metrics: language model, entailment model, language complexity
python3 DE_main.py \
        --input_file  /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.entail.txt \
        --input_augmented_pkl_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.pkl \
        --output_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/train.qa.0_10000.entail.de.txt \
        &>> experiments_3_repeat_da_de

python3 DE_main.py \
        --input_file  /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.entail.txt \
        --input_augmented_pkl_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.pkl \
        --output_file /scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/wiki10000.qa.0_10000.entail.de.txt \
        &>> experiments_3_repeat_da_de
