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


# squad data
input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/"
output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
data_type="squad"
data_file_prefix="train"
st_idx=0
ed_idx=10000
sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt"


# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/SQuAD2.0/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/SQuAD2.0/"
# data_type="squad"
# data_file_prefix="train"
# st_idx=50000
# ed_idx=92210
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# wiki data
input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=0
ed_idx=10000
sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt"



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=50000
# ed_idx=100000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=100000
# ed_idx=150000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=150000
# ed_idx=200000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=200000
# ed_idx=250000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=250000
# ed_idx=300000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=300000
# ed_idx=350000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 


# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=350000
# ed_idx=400000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=400000
# ed_idx=450000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=450000
# ed_idx=500000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=500000
# ed_idx=550000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=550000
# ed_idx=600000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=600000
# ed_idx=650000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=650000
# ed_idx=700000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=700000
# ed_idx=750000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=750000
# ed_idx=800000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=800000
# ed_idx=850000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 



# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=850000
# ed_idx=900000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 




# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=900000
# ed_idx=950000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 


# input_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/original/Wiki10000/"
# output_path="/scratch/scratch8/madhurjindal/ACS-QG-Scratch/Datasets/processed/Wiki10000/"
# data_type="wiki10000"
# data_file_prefix="wiki10000"
# st_idx=950000
# ed_idx=1000000
# sort "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.txt" | uniq  > "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.txt" 
