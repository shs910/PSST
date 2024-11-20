BASE_PATH="/home/yzyang/LLM/shs/code/PSST"
LAUNCH_FILE_PATH="${BASE_PATH}/style_evaluation/passage_segmentasion_baseline_paraphrase.py"
export CUDA_VISIBLE_DEVICES=0
export STANZA_RESOURCES_DIR="/home/yzyang/stanza_resources"

EXP_DIR="${BASE_PATH}/oral_exp"

# data
LEVEL="token_num"
DATA_DIR_PATH="${EXP_DIR}/1_different_level/${LEVEL}/raw"

TOKEN_NUM_LIST=("800_delta_200" "400_delta_100" "1200_delta_200")
# TOKEN_NUM_LIST=("1200_delta_200")

# runtime/home/shs_910/sunhuashan/code/PSST/style_evaluation/sh_baseline_by_token/passage_segmentation.sh
DATA_NAME_LIST=("gpt-3.5_paraphrase")
# DATA_NAME_LIST=("Llama-2-7b-chat-ms" "Llama-2-70B-Chat-GPTQ" "Llama-3-8B-Instruct")
for TOKEN_NUM in "${TOKEN_NUM_LIST[@]}";do
    for DATA_NAME in "${DATA_NAME_LIST[@]}";do
        echo "${TOKEN_NUM}-${DATA_NAME}"
        SRC_DATA_PATH="${DATA_DIR_PATH}/${TOKEN_NUM}/data/${DATA_NAME}.jsonl"
        OUTPUT_FILE_PATH="${DATA_DIR_PATH}/${TOKEN_NUM}/segmentation/segmentation_${DATA_NAME}.jsonl"
        python $LAUNCH_FILE_PATH \
        --src_data_path $SRC_DATA_PATH \
        --output_path $OUTPUT_FILE_PATH
    done
done

