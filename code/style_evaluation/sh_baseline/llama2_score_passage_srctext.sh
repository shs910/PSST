BASE_PATH="/home/yzyang/LLM/shs/code/PSST"
LAUNCH_FILE_PATH="${BASE_PATH}/style_evaluation/scoring_baseline.py"
export CUDA_VISIBLE_DEVICES=7

EXP_DIR="${BASE_PATH}/oral_exp"

# data
LEVEL="token_num"
DATA_DIR_PATH="${EXP_DIR}/1_different_level/${LEVEL}/raw"

TOKEN_NUM_LIST=("800_delta_200" "400_delta_100" "1200_delta_200")
# TOKEN_NUM_LIST=(60)

# score model
SCORE_MODEL_DIR_PATH="${BASE_PATH}/data/for_model_traning"

TGT_DIM="emotionality"
TRANING_TYPE="emotionality_gpt-3.5_with_out_0"

SCORE_MODEL_TYPE="llama2"
SCORE_MODEL_NAME="llama_1.1b"
SCORE_MODEL_NAME_PATH="${SCORE_MODEL_DIR_PATH}/${TRANING_TYPE}/model_output/${SCORE_MODEL_NAME}"

# runtime
# TGT_DIM="interactivaty,vividness"
SCORE_MODE=("per_1,per_2,per_3,per_4")
DATA_NAME_LIST=("src_text" "ted_text")
# DATA_NAME_LIST=("ted_text")
# DATA_NAME_LIST=("src_text")

for TOKEN_NUM in "${TOKEN_NUM_LIST[@]}";do
    for DATA_NAME in "${DATA_NAME_LIST[@]}";do
        echo "${TOKEN_NUM}-${DATA_NAME}"
        SRC_DATA_PATH="${DATA_DIR_PATH}/${TOKEN_NUM}/segmentation/segmentation_${DATA_NAME}.jsonl"
        OUTPUT_FILE_PATH="${DATA_DIR_PATH}/${TOKEN_NUM}/scored/${TRANING_TYPE}-${SCORE_MODEL_TYPE}-${SCORE_MODEL_NAME}/scored_${DATA_NAME}.jsonl"
        python $LAUNCH_FILE_PATH \
        --src_data_path $SRC_DATA_PATH \
        --output_path $OUTPUT_FILE_PATH \
        --tgt_dim_list $TGT_DIM \
        --score_mode $SCORE_MODE \
        --used_model $SCORE_MODEL_TYPE \
        --used_model_path $SCORE_MODEL_NAME_PATH
    done
done