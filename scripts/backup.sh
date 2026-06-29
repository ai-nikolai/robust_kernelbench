# USAGE
# This script is inteneded to be used as a tool to clean the experiments folder, or to copy the experiments folder for a potential backup

# USEFUL COMMAND
# zip -r experiments_backup_slurm_ukp_09_04_2026.zip experiments_backup2
# zip -r experiments_backup_runpod_10_04_2026.zip experiments_backup3
# zip -r experiments_all_submission_06_05_2026.zip experiments_backup2
# unsip xyz.zip


# Scripts to clean eval & compilations.csv
    # "exp_local_L1_V8_3_deepseek_r1_0528__API1"
    # "exp_local_L1_V8_3_glm_5__API1"
    # "exp_local_L1_V8_3_qwen3_5_397b_a17b__API1"
    # "exp_local_L1_V8_3_qwen3_coder__API1"

# RUNPOD
experiments=(
    "exp_local_L1_V8_3_deepseek_v3_1_terminus__API1"
    "exp_local_L1_V8_3_devstral_2512__API1"
    "exp_local_L1_V8_3_gpt_oss_120b__API1"
)

# SLURM CLUSTER
# experiments=(
#     "exp_local_L1_V8_3_1_Qwen3_30B_A3B_Instruct_2507__API0"
#     "exp_local_L1_V8_3_1_Qwen3_Coder_30B_A3B_Instruct__API0"
#     "exp_local_L1_V8_3_1_Qwen3_4B__API0"
#     "exp_local_L1_V8_3_1_Qwen3_8B__API0"
#     "exp_local_L1_V8_3_1_Qwen3_14B__API0"
#     "exp_local_L1_V8_3_1_Qwen3_32B__API0"
#     "exp_local_L1_V8_3_1_DeepSeek_Coder_V2_Lite_Instruct__API0"
# )

trials=(
    "1"
    "2"
    "3"
    "12"
    "13"
    "22"
    "23"
)

# trials=(
#     "1"
#     "4"
#     "7"
#     "204"
#     "207"
#     "304"
#     "307"
# )

# EXPERIMENT_NAME="exp_local_v6L_2_Qwen3_Coder_30B_A3B_Instruct_kernelbench_20260203_184734"
# TRIAL=${1:-"None"} #pass a param

# BASE_FOLDER="experiments"
BASE_FOLDER="experiments_backup2"


## COPY COMMAND
echo "BACKING Up...."
for experiment in "${experiments[@]}"; do
    read -r experiment_name <<< "$experiment"
    for trial in "${trials[@]}"; do
        read -r trial_num <<< "$trial"
        # RUNNING THE ACTUAL COMMAND
        # rm ./${BASE_FOLDER}/${experiment_name}/trial_${trial_num}/compilations.csv
        # rm ./${BASE_FOLDER}/${experiment_name}/trial_${trial_num}/evaluations.csv

        folder_path="./${BASE_FOLDER}/${experiment_name}/trial_${trial_num}/"
        original_folder_path="./experiments/${experiment_name}/trial_${trial_num}"
        echo "Copying: $folder_path"
        mkdir -p $folder_path
        mkdir -p "$folder_path/code"

        cp "${original_folder_path}/metadata.json" "${folder_path}/"
        cp "${original_folder_path}/generations.csv" "${folder_path}/"
        cp "${original_folder_path}/compilations.csv" "${folder_path}/"
        cp "${original_folder_path}/evaluations.csv" "${folder_path}/"

        cp -r "${original_folder_path}/code/" "${folder_path}/"


    done
done


## DELETE COMMAND
# for experiment in "${experiments[@]}"; do
#     read -r experiment_name <<< "$experiment"
#     for trial in "${trials[@]}"; do
#         read -r trial_num <<< "$trial"
#         # RUNNING THE ACTUAL COMMAND
#         # rm ./${BASE_FOLDER}/${experiment_name}/trial_${trial_num}/compilations.csv
#         # rm ./${BASE_FOLDER}/${experiment_name}/trial_${trial_num}/evaluations.csv

#         folder_path="./${BASE_FOLDER}/${experiment_name}/trial_${trial_num}/kernel/"
#         echo $folder_path
#         rm -rf $folder_path
#     done
# done