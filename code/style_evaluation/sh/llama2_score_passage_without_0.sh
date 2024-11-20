BASE_PATH="/home/shs_910/sunhuashan/code/PSST"
LAUNCH_FILE_PATH="${BASE_PATH}/style_evaluation/scoring.py"
export CUDA_VISIBLE_DEVICES=0

EXP_DIR="${BASE_PATH}/oral_exp"

# data
DATA_DIR_PATH="${EXP_DIR}/test_res"

# score model
SCORE_MODEL_DIR_PATH="${BASE_PATH}/data/for_model_traning"
TRANING_TYPE="interactivity_gpt-3.5_with_out_0"
SCORE_MODEL_TYPE="llama2"
SCORE_MODEL_NAME="llama_1.1b"
SCORE_MODEL_NAME_PATH="${SCORE_MODEL_DIR_PATH}/${TRANING_TYPE}/model_output/${SCORE_MODEL_NAME}"

# runtime
# TGT_DIM="interactivaty,vividness"
TGT_DIM="interactivity"
SCORE_MODE=("per_1,per_2,per_3,per_4")
# MODEL_NAME_LIST=("gpt-3.5" "gpt-3.5-paraphrase" "llama_2_chat_13b" "Llama-2-7b-chat-ms")
# MODEL_NAME_LIST=("Llama-3-8B-Instruct"  "vicuna-7b-v1.5" "vicuna-13b-v1.5")
MODEL_NAME_LIST=("Llama-2-70B-Chat-GPTQ")
DATA_NAME_LIST=("with_concise_prompt" "with_enhanced_prompt")
# DATA_NAME_LIST=("with_concise_prompt")

for DATA_NAME in "${DATA_NAME_LIST[@]}";do
    for MODEL_NAME in "${MODEL_NAME_LIST[@]}";do
        echo "${DATA_NAME}-${MODEL_NAME}"
        SRC_DATA_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/segmentation/segmentation_${DATA_NAME}.jsonl"
        OUTPUT_FILE_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/scored/${TRANING_TYPE}-${SCORE_MODEL_TYPE}-${SCORE_MODEL_NAME}/scored_${DATA_NAME}.jsonl"
        python $LAUNCH_FILE_PATH \
        --src_data_path $SRC_DATA_PATH \
        --output_path $OUTPUT_FILE_PATH \
        --tgt_dim_list $TGT_DIM \
        --score_mode $SCORE_MODE \
        --used_model $SCORE_MODEL_TYPE \
        --used_model_path $SCORE_MODEL_NAME_PATH
    done
done