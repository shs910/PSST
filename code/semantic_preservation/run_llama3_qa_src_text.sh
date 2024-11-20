export CUDA_VISIBLE_DEVICES=4,5
# 1. custom your path
BASE_PATH="/home/yzyang/LLM/shs/code/PSST"
MODEL_DIR_PATH="/home/yzyang/LLM/shs/models/llama3"
QA_MODEL_NAME_LIST=("Llama-3-8B-Instruct")

LAUNCH_FILE_PATH="${BASE_PATH}/semantic_preservation/qa_based/QA_Inference.py"

EXP_DIR="${BASE_PATH}/oral_exp/qa_eval"
# data
DATA_DIR_PATH="${EXP_DIR}/qa_pairs"
OUTPUT_DIR_PATH="${EXP_DIR}/qa_res"

TEST_MODEL_LIST=("baseline")
BATCH_SIZE=4
NUM_LIST=(800 400 1200)
for NUM in "${NUM_LIST[@]}";do
    TEST_DATA_NAME_LIST=("${NUM}_sample_QA1" "${NUM}_sample_QA2")
    for QA_MODEL in "${QA_MODEL_NAME_LIST[@]}";do
        for TEST_MODEL_NAME in "${TEST_MODEL_LIST[@]}";do
            for DATA_NAME in "${TEST_DATA_NAME_LIST[@]}";do
                MODEL_NAME_OR_PATH=${MODEL_DIR_PATH}/${QA_MODEL}
                DATA_FILE_PATH=${DATA_DIR_PATH}/${TEST_MODEL_NAME}/${DATA_NAME}.jsonl
                OUTPUT_FILE_PATH=${OUTPUT_DIR_PATH}/${TEST_MODEL_NAME}/${QA_MODEL}/${DATA_NAME}.jsonl
                python $LAUNCH_FILE_PATH \
                --src_data_path ${DATA_FILE_PATH} \
                --output_path ${OUTPUT_FILE_PATH} \
                --model_path ${MODEL_NAME_OR_PATH} \
                --batch_size ${BATCH_SIZE}
            done
        done
    done
done