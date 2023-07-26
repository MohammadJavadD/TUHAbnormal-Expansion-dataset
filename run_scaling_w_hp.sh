#!/bin/bash
#SBATCH --job-name=run_NMT_scaling_hps_pp3
#SBATCH --output=./logs/NMT_scaling_hps_pp3_%A_%a.out
#SBATCH --error=./logs/NMT_scaling_hps_pp3_%A_%a.err
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000M
#SBATCH --time=12:00:00
#SBATCH --wait


module load anaconda/3
conda activate braindecode
# source ~/braindecode3/bin/activate
# Define an array of numbers
# models=('TCN' 'Deep4Net' 'Shallow' 'EEGNet')
models=('Deep4Net')
seeds=(10 100 1000)
# numbers=(2100)
numbers=(25 40 55 70 85 100 500 1000 1500 2100)
# numbers=(2700)


# Define an array of hps
lrs=(0.005 0.001 0.0001)
# weight_decays=(5e-3 5e-5 5e-7)
weight_decays=(5e-5)
batch_sizes=(16 64 256)
# drop_prob=(1e-1 1e-2 1e-3)
drop_prob=(1e-3)


echo $SLURM_ARRAY_TASK_ID

# Run your Python script with the picked number as an argument
cs=()
# index=10
index=$SLURM_ARRAY_TASK_ID
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
tar -xf ~/scratch/medical/eeg/tuab/tuab_pp3.tar -C $SLURM_TMPDIR
tar -xf ~/scratch/medical/eeg/NMT/nmt_pp3.tar -C $SLURM_TMPDIR


# Merge 
for n in "${numbers[@]}"; do
  python main.py \
      --task_name NMT_deep4_hps_pp3_wN_wAug_WoPt_index${index}_number${n}\
      --model_name ${comb[0]}\
      --seed ${comb[1]}\
      --lr ${comb[2]}\
      --weight_decay ${comb[3]}\
      --batch_size ${comb[4]}\
      --drop_prob ${comb[5]}\
      --train_folder $SLURM_TMPDIR/tuab_pp3\
      --train_folder2 $SLURM_TMPDIR/nmt_pp3\
      --result_folder ~/scratch/medical/eeg/results_pp3\
      --ids_to_load_train 0\
      --ids_to_load_train2 ${n}\
      --n_epochs 35\
      --augment True\
      # --pre_trained True\
      # --load_path '/home/mila/m/mohammad-javad.darvishi-bayasi/scratch/medical/eeg/results_pp3/Deep4Net/seed10/TUH_deep4_hps_pp3_wN_wAug_index9_number2700/Deep4Net_trained_TUH_deep4_hps_pp3_wN_wAug_index9_number2700_state_dict_10.pt'
    done

# --ids_to_load_train2 ${comb[2]}\
