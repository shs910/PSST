# 1. custom your path
BASE_PATH="/home/yzyang/LLM/shs/opensource_your_work/PSST/code"
MODEL_DIR_PATH="/home/yzyang/LLM/shs/models/llama2"

LAUNCH_FILE_PATH="${BASE_PATH}/style_transfer/pipeline_inference.py"
EXP_DIR="${BASE_PATH}/oral_exp"

# 2. choose the testing data
TOKEN_NUM_LIST=("1200_delta_200")
DATA_DIR_PATH="${EXP_DIR}/raw"
OUTPUT_DIR_PATH="${EXP_DIR}/test_res"

# 3. choose your model
# "Llama-2-13b-chat-ms" "Llama-2-70B-Chat"
MODEL_NAME_LIST=("Llama-2-13b-chat-ms")
DATA_NAME_LIST=("with_concise_prompt" "with_enhanced_prompt")

BATCH_SIZE=1
for TOKEN_NUM in "${TOKEN_NUM_LIST[@]}";do
    for DATA_NAME in "${DATA_NAME_LIST[@]}";do
        for MODEL_NAME in "${MODEL_NAME_LIST[@]}";do
            MODEL_NAME_OR_PATH=${MODEL_DIR_PATH}/${MODEL_NAME}
            DATA_FILE_PATH=${DATA_DIR_PATH}/${TOKEN_NUM}/data/src_text_${DATA_NAME}.jsonl
            OUTPUT_FILE_PATH=${OUTPUT_DIR_PATH}/${MODEL_NAME}/prediction/prediction_sentence_${TOKEN_NUM}_${DATA_NAME}.jsonl
            python $LAUNCH_FILE_PATH \
            --src_data_path ${DATA_FILE_PATH} \
            --output_path ${OUTPUT_FILE_PATH} \
            --model_path ${MODEL_NAME_OR_PATH} \
            --batch_size ${BATCH_SIZE}
        done
    done
done