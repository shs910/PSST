# 1. custom your path
BASE_PATH="/home/yzyang/LLM/shs/opensource_your_work/PSST/code"

LAUNCH_FILE_PATH="${BASE_PATH}/style_transfer/api_inference.py"
EXP_DIR="${BASE_PATH}/oral_exp"

# 2. choose the testing data
# ("800_delta_200" "400_delta_100")
TOKEN_NUM_LIST=("1200_delta_200")
DATA_DIR_PATH="${EXP_DIR}/raw"
OUTPUT_DIR_PATH="${EXP_DIR}/test_res"
DATA_NAME_LIST=("with_concise_prompt"  "with_enhanced_prompt_v3")

# 3. choose your model
MODEL_NAME_LIST=("gpt-3.5")

for TOKEN_NUM in "${TOKEN_NUM_LIST[@]}";do
    for DATA_NAME in "${DATA_NAME_LIST[@]}";do
        for MODEL_NAME in "${MODEL_NAME_LIST[@]}";do
            DATA_FILE_PATH=${DATA_DIR_PATH}/${TOKEN_NUM}/data/src_text_${DATA_NAME}.jsonl
            OUTPUT_FILE_PATH=${OUTPUT_DIR_PATH}/${MODEL_NAME}/prediction/prediction_sentence_${TOKEN_NUM}_${DATA_NAME}.jsonl
            python $LAUNCH_FILE_PATH \
            --src_data_path ${DATA_FILE_PATH} \
            --output_path ${OUTPUT_FILE_PATH} \
            --model_name ${MODEL_NAME}
        done
    done
done