BASE_PATH="/home/yzyang/LLM/shs/code/PSST"
LAUNCH_FILE_PATH="${BASE_PATH}/style_evaluation/passage_segmentasion.py"
export CUDA_VISIBLE_DEVICES=0
export STANZA_RESOURCES_DIR="/home/yzyang/stanza_resources"

EXP_DIR="${BASE_PATH}/oral_exp"

# data
LEVEL="token_num"
DATA_DIR_PATH="${EXP_DIR}/1_different_level/${LEVEL}/test_res"

# TOKEN_NUM_LIST=("400_delta_100" "800_delta_200")
TOKEN_NUM_LIST=("1200_delta_200")

# runtime
DATA_NAME_LIST=("with_enhanced_prompt_v3" "with_concise_prompt")
# DATA_NAME_LIST=("with_enhanced_prompt_v3")
# MODEL_NAME_LIST=("gpt-3.5" "Llama-2-7b-chat-ms" "Llama-2-13b-chat-ms" "Llama-3-8B-Instruct")
MODEL_NAME_LIST=("Llama-2-70B-Chat-GPTQ" "Llama-3-70B-Instruct-GPTQ")
# DATA_NAME_LIST=("src_text" "ted_text")
for TOKEN_NUM in "${TOKEN_NUM_LIST[@]}";do
    for MODEL_NAME in "${MODEL_NAME_LIST[@]}";do
        for DATA_NAME in "${DATA_NAME_LIST[@]}";do
            echo "${TOKEN_NUM}-${MODEL_NAME}-${DATA_NAME}"
            SRC_DATA_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/prediction/prediction_sentence_${TOKEN_NUM}_${DATA_NAME}.jsonl"
            OUTPUT_FILE_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/segmentation/segmentation_sentence_${TOKEN_NUM}_${DATA_NAME}.jsonl"
            python $LAUNCH_FILE_PATH \
            --src_data_path $SRC_DATA_PATH \
            --output_path $OUTPUT_FILE_PATH
        done
    done
done