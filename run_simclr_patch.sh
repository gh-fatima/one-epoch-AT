#!/bin/bash

> test_results_simclr_patch_0.75.txt

# Define arrays of parameter values
test_patches_values=("1" "32" "64")

model_path_values=("/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8_patch_0.75vs0.25/99.pt"
"/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8_patch_0.75vs0.25/199.pt"
"/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8_patch_0.75vs0.25/299.pt"
"/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8_patch_0.75vs0.25/399.pt"
"/home/Fatemeh/One-epoch/EMP-SSL-main/logs/SimCLR-Training/SimCLR_bs256_NONE_adv/save_models_adv_wo_Normalization_8_patch_0.75vs0.25/499.pt")

scale_min=0.75
scale_max=0.75
ratio_min=1
ratio_max=1
type="patch"

# Loop through the parameter combinations and run the code
for test_patches in "${test_patches_values[@]}"; do
    for model_path in "${model_path_values[@]}"; do
            echo "Running with test_patches = $test_patches, model_path = $model_path, type = $type, scale_min = $scale_min, scale_max = $scale_max, ratio_min = $ratio_min, ratio_max = $ratio_max"
            python evaluate_simclr.py --test_patches "$test_patches" --model_path "$model_path" --type "$type" --scale_min "$scale_min" --scale_max "$scale_max" --ratio_min "$ratio_min" --ratio_max "$ratio_max" >> test_results_simclr_patch_0.75.txt

    done
done
echo "Results written to test_results_simclr_patch_0.75.txt"
