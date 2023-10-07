#!/bin/bash

> test_results_EMP_patch_knn_patch.txt


# Define arrays of parameter values
test_patches_values=("1" "32" "40" "64")

model_path_values=("/home/Fatemeh/One-epoch/EMP-SSL-main/logs/EMP-SSL-Training/patchsim200_numpatch40_bs100_lr0.3_NONE/save_models_adv_wo_Normalization_8/29.pt")

scale_min=0.25
scale_max=0.25
ratio_min=1
ratio_max=1
num_class=10
type="patch"
data="cifar10"

# Loop through the parameter combinations and run the code
for test_patches in "${test_patches_values[@]}"; do
    for model_path in "${model_path_values[@]}"; do
            echo "Running with test_patches = $test_patches, model_path = $model_path, type = $type, scale_min = $scale_min, scale_max = $scale_max, ratio_min = $ratio_min, ratio_max = $ratio_max"
            python evaluate_2_knn.py --num_class "$num_class" --data "$data" --test_patches "$test_patches" --model_path "$model_path" --type "$type" --scale_min "$scale_min" --scale_max "$scale_max" --ratio_min "$ratio_min" --ratio_max "$ratio_max" >> test_results_EMP_patch_knn_patch.txt


    done
done
echo test_results_EMP_patch_knn_patch.txt