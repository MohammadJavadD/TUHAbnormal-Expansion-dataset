#!/bin/bash
#SBATCH --job-name=run_TUH_scaling_hps_pp6
#SBATCH --output=./logs/TUH_scaling_hps_pp6_%A_%a.out
#SBATCH --error=./logs/TUH_scaling_hps_pp6_%A_%a.err
#SBATCH --array=0-28
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000M
#SBATCH --time=2:00:00
#SBATCH --wait


module load anaconda/3
conda activate braindecode
# source ~/braindecode3/bin/activate
# Define an array of numbers
# models=('TCN' 'Deep4Net' 'Shallow' 'EEGNet')
models=('Deep4Net')
seeds=(10 100 1000)
# numbers=(2100)
# numbers=(25 40 55 70 85 100 500 1000 1500 2100)
numbers=(2700)


# Define an array of hps
lrs=(0.005 0.001 0.0001)
weight_decays=(5e-3 5e-5 5e-7)
batch_sizes=(16 64 256)

echo $SLURM_ARRAY_TASK_ID

# Run your Python script with the picked number as an argument
cs=()
index=0
# $SLURM_ARRAY_TASK_ID
# +2993
counter=0
for a in "${models[@]}"; do
  for b in "${seeds[@]}"; do
    # for d in "${numbers[@]}"; do
      for e in "${lrs[@]}"; do
        for f in "${weight_decays[@]}"; do
          for g in "${batch_sizes[@]}"; do
            if ((counter == ${index})); then
                comb+=($a)
                comb+=($b)
                comb+=($c)
                comb+=($d)
                comb+=($e)
                comb+=($f)
                comb+=($g)
            fi
            ((counter++))
          done
        done
      done
    # done
  done
done


echo "Selected element: ${comb[@]}"

# Merge 
for d in "${numbers[@]}"; do
  python main.py \
      --task_name TUH_deep4_hps_index${index}_number${d}\
      --model_name ${comb[0]}\
      --seed ${comb[1]}\
      --train_folder '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/tuab/tuab_pp6'\
      --train_folder2 '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/NMT/nmt_pp6'\
      --result_folder '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp_6/'\
      --ids_to_load_train 0\
      --ids_to_load_train2 ${d}\
      --lr ${comb[2]}\
      --weight_decay ${comb[3]}\
      --batch_size ${comb[4]}\
      --augment True
done

# --ids_to_load_train2 ${comb[2]}\
