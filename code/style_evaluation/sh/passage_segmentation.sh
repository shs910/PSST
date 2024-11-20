BASE_PATH="/home/shs_910/sunhuashan/code/PSST"
LAUNCH_FILE_PATH="${BASE_PATH}/style_evaluation/passage_segmentasion.py"
export CUDA_VISIBLE_DEVICES=0
export STANZA_RESOURCES_DIR="/home/yzyang/stanza_resources"

EXP_DIR="${BASE_PATH}/oral_exp"

# data
DATA_DIR_PATH="${EXP_DIR}/test_res"

# runtime
# MODEL_NAME_LIST=("gpt-3.5" "gpt-3.5-paraphrase" "llama_2_chat_13b" "Llama-2-7b-chat-ms")
# MODEL_NAME_LIST=("alpaca-7b-native" "gpt-3.5" "gpt-3.5-paraphrase" "llama_2_chat_13b" "Llama-2-7b-chat-ms" "Llama-3-8B-Instruct" "ted_data" "ted_data_v2" "vicuna-7b-v1.5" "vicuna-13b-v1.5")
MODEL_NAME_LIST=("Llama-2-70B-Chat-GPTQ")
# MODEL_NAME_LIST=("alpaca-7b-native" "vicuna-7b-v1.5" "Llama-2-7b-chat-ms" "Llama-3-8B-Instruct")
DATA_NAME_LIST=("with_concise_prompt" "with_enhanced_prompt")

for DATA_NAME in "${DATA_NAME_LIST[@]}";do
    for MODEL_NAME in "${MODEL_NAME_LIST[@]}";do
        echo "${DATA_NAME}-${MODEL_NAME}"
        SRC_DATA_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/prediction/prediction_${DATA_NAME}.jsonl"
        OUTPUT_FILE_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/segmentation/segmentation_${DATA_NAME}.jsonl"
        python $LAUNCH_FILE_PATH \
        --src_data_path $SRC_DATA_PATH \
        --output_path $OUTPUT_FILE_PATH
    done
done

