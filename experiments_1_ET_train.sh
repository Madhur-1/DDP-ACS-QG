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

python3 -u run_glue.py \
        --model_type xlnet \
        --model_name_or_path /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/models/xlnet-base-cased \
        --task_name MRPC \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/glue_data/squad-rte/MRPC \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /scratch/scratch8/madhurjindal/ACS-QG-Scratch/ET/et_outdir/xlnet-base-cased \
        --overwrite_output_dir \
        &> experiments_1_ET_train