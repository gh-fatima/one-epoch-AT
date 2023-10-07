#!/bin/bash

> test_results_simclr_patch_0.75_cifar100.txt

# Define arrays of parameter values
test_patches_values=("1" "32" "64")
# test_patches_values=("1")
#"/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8/99.pt"
#    "/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8/199.pt"
#"/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8/299.pt" 

model_path_values=("/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8_patch_0.75vs0.25_cifar100/499.pt")

data="cifar100"

# Loop through the parameter combinations and run the code
for test_patches in "${test_patches_values[@]}"; do
    for model_path in "${model_path_values[@]}"; do
        type="patch"
        echo "Running with test_patches = $test_patches, model_path = $model_path, type = $type"
        python evaluate_simclr.py --data "$data" --test_patches "$test_patches" --model_path "$model_path" --type "$type" >> test_results_simclr_patch_0.75_cifar100.txt
        # done
    done
done

echo "test_results_simclr_patch_0.75_cifar100.txt"
