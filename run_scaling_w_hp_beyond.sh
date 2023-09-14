#!/bin/bash
#SBATCH --job-name=run_Merge_scaling_hps_pp3
#SBATCH --output=./logs/Merge_scaling_hps_pp3_%A_%a.out
#SBATCH --error=./logs/Merge_scaling_hps_pp3_%A_%a.err
#SBATCH --array=1,2,3,4,5,8,17
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=48000M
#SBATCH --time=4:00:00
#SBATCH --wait


# module load anaconda/3
conda activate braindecode
# source ~/braindecode3/bin/activate

wandb online

# Define an array of numbers
models=('TCN' 'Deep4Net' 'Shallow' 'EEGNet')
# models=('Deep4Net')
# seeds=(10 100 1000)
seeds=(2023 2024 10 100 1000)
numbers=(2100)
# numbers=(20 25 40 55 70 85 100 250 800 1000 2700)
# numbers=(20 25 31 40 52 66 85 109 140 179 229 293 375 481 615 787 1007 1289 1649 2110 2700)
# # numbers=(20)
# numbers=(50 50)


# Define an array of hps
# lrs=(0.005 0.001 0.0001)
lrs=(0.0001)
# weight_decays=(5e-3 5e-5 5e-7)
weight_decays=(5e-5)
# batch_sizes=(16 64 256)
batch_sizes=(64)
# drop_prob=(1e-1 1e-2 1e-3)
drop_prob=(1e-3)


echo $SLURM_ARRAY_TASK_ID

# Run your Python script with the picked number as an argument
cs=()
index=1
# index=$SLURM_ARRAY_TASK_ID
# [Note]
# +2993
counter=0
for a in "${models[@]}"; do
  for b in "${seeds[@]}"; do
 for c in "${lrs[@]}"; do
for d in "${weight_decays[@]}"; do
  for e in "${batch_sizes[@]}"; do
 for f in "${drop_prob[@]}"; do

if ((counter == ${index})); then
 comb+=($a)
 comb+=($b)
 comb+=($c)
 comb+=($d)
 comb+=($e)
 comb+=($f)
 # comb+=($g)
fi
((counter++))
 done
  done
done
 done
  done
done


echo "Selected element: ${comb[@]}"

## extract datasets to $slurm_tmpdir
# # tar -cf dataset.tar dataset/
# tar -xf ~/scratch/medical/eeg/tuab/tuab_pp3.tar -C $SLURM_TMPDIR
# tar -xf ~/scratch/medical/eeg/NMT/nmt_pp3.tar -C $SLURM_TMPDIR


# Merge 
for n in "${numbers[@]}"; do
  python main_beyond.py\
  --task_name rnd5_mila_Allmodels_merge_full_wN_WAug_DefArgs_index${index}_number${n}\
  --model_name ${comb[0]}\
  --seed ${comb[1]}\
  --lr ${comb[2]}\
  --weight_decay ${comb[3]}\
  --batch_size ${comb[4]}\
  --drop_prob ${comb[5]}\
  --train_folder $SLURM_TMPDIR/tuab_pp3\
  --train_folder2 $SLURM_TMPDIR/nmt_pp3\
  --result_folder ~/scratch/medical/eeg/results_pp3/\
  --ids_to_load_train 2700\
  --ids_to_load_train2 ${n}\
  --n_epochs 35\
  --use_defualt_parser True\
  --augment True
  # --pre_trained True\
  # --load_path /home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/TCN/seed100/tuh_scaling_wN_WAug_DefArgs_index3_number2700/TCN_trained_tuh_scaling_wN_WAug_DefArgs_index3_number2700_state_dict_100.pt
  # --load_path /home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/EEGNet/seed1000/tuh_scaling_wN_WoAug_DefArgs_index19_number2700/EEGNet_trained_tuh_scaling_wN_WoAug_DefArgs_index19_number2700_state_dict_1000.pt
  # --load_path /home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/Shallow/seed1000/tuh_scaling_wN_WAug_DefArgs_index14_number2700/Shallow_trained_tuh_scaling_wN_WAug_DefArgs_index14_number2700_state_dict_1000.pt
  # --load_path '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/TCN/seed100/tuh_scaling_wN_WAug_DefArgs_index3_number2700/TCN_trained_tuh_scaling_wN_WAug_DefArgs_index3_number2700_state_dict_100.pt'
  # --load_path '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/Deep4Net/seed100/tuh_scaling_wN_WAug_DefArgs_index8_number2700/Deep4Net_trained_tuh_scaling_wN_WAug_DefArgs_index8_number2700_state_dict_100.pt'
# --load_path '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results/Deep4Net/seed2024/Deep4Net_trained_tuh_pp3_WoN_wAug_W0pt_Womix10/Deep4Net_trained_Deep4Net_trained_tuh_pp3_WoN_wAug_W0pt_Womix10_state_dict_2024.pt'

# python main.py\
#   --task_name ft20_mila_Deep4Net_NMT_scaling_woN_WAug_DefArgs_index${index}_number${n}\
#   --model_name ${comb[0]}\
#   --seed ${comb[1]}\
#   --lr ${comb[2]}\
#   --weight_decay ${comb[3]}\
#   --batch_size ${comb[4]}\
#   --drop_prob ${comb[5]}\
#   --train_folder $SLURM_TMPDIR/tuab_pp3\
#   --train_folder2 $SLURM_TMPDIR/nmt_pp3\
#   --result_folder ~/scratch/medical/eeg/results_pp3/\
#   --ids_to_load_train 0\
#   --ids_to_load_train2 ${n}\
#   --n_epochs 35\
#   --use_defualt_parser True\
#   --augment True\
#   --pre_trained True\
#   --load_path /home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results/Deep4Net/seed2024/Deep4Net_trained_tuh_pp3_WoN_wAug_W0pt_Womix10/Deep4Net_trained_Deep4Net_trained_tuh_pp3_WoN_wAug_W0pt_Womix10_state_dict_2024.pt
#   # --load_path ~/scratch/medical/eeg/results_pp3/Deep4Net/seed100/tuh_scaling_wN_WAug_DefArgs_index8_number2700/Deep4Net_trained_tuh_scaling_wN_WAug_DefArgs_index8_number2700_state_dict_100.pt

#   # --load_path /home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/TCN/seed100/tuh_scaling_wN_WAug_DefArgs_index3_number2700/TCN_trained_tuh_scaling_wN_WAug_DefArgs_index3_number2700_state_dict_100.pt

#   # --pre_trained True\
#   # --load_path /home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/EEGNet/seed1000/tuh_scaling_wN_WoAug_DefArgs_index19_number2700/EEGNet_trained_tuh_scaling_wN_WoAug_DefArgs_index19_number2700_state_dict_1000.pt
 
done
# 
# mixup_sm30_fsh30_merge
# --ids_to_load_train2 ${comb[2]}\
# mixcls_lr0001_Aug10_merge


