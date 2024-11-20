BASE_PATH="/home/yzyang/LLM/shs/code/PSST"
LAUNCH_FILE_PATH="${BASE_PATH}/style_evaluation/scoring.py"
export CUDA_VISIBLE_DEVICES=1

EXP_DIR="${BASE_PATH}/oral_exp"

# data
LEVEL="token_num"
DATA_DIR_PATH="${EXP_DIR}/1_different_level/${LEVEL}/test_res"

# TOKEN_NUM_LIST=(20 40 60)
# TOKEN_NUM_LIST=("800_delta_200" "400_delta_100" )
TOKEN_NUM_LIST=("1200_delta_200")
# runtime
# TGT_DIM="interactivaty,vividness"

SCORE_MODE=("per_1,per_2,per_3,per_4")

# runtime
DATA_NAME_LIST=("with_concise_prompt" "with_enhanced_prompt_v3")
# DATA_NAME_LIST=("with_enhanced_prompt_v3")

# MODEL_NAME_LIST=("Llama-2-7b-chat-ms" "Llama-2-13b-chat-ms" "Llama-3-8B-Instruct" "vicuna-7b-1v5")
# MODEL_NAME_LIST=("Llama-2-70B-Chat-GPTQ" "Llama-3-8B-Instruct")
# MODEL_NAME_LIST=("gpt-3.5" "Llama-2-7b-chat-ms" "Llama-2-13b-chat-ms" "Llama-3-8B-Instruct")
# MODEL_NAME_LIST=("Llama-2-70B-Chat-GPTQ" "Llama-3-70B-Instruct-GPTQ")
MODEL_NAME_LIST=("Llama-3-70B-Instruct-GPTQ")
# score model
SCORE_MODEL_DIR_PATH="${BASE_PATH}/data/for_model_traning"
SCORE_MODEL_TYPE="llama2"
SCORE_MODEL_NAME="llama_1.1b"
# TGT_DIM_LIST=("emotionality" "orality" "vividness" "interactivity")
TGT_DIM_LIST=("interactivity")
for TGT_DIM in "${TGT_DIM_LIST[@]}";do
    TRANING_TYPE="${TGT_DIM}_gpt-3.5_with_out_0"
    SCORE_MODEL_NAME_PATH="${SCORE_MODEL_DIR_PATH}/${TRANING_TYPE}/model_output/${SCORE_MODEL_NAME}"
    for TOKEN_NUM in "${TOKEN_NUM_LIST[@]}";do
        for MODEL_NAME in "${MODEL_NAME_LIST[@]}";do
            for DATA_NAME in "${DATA_NAME_LIST[@]}";do
                echo "${DATA_NAME}-${MODEL_NAME}-${TOKEN_NUM}"
                SRC_DATA_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/segmentation/segmentation_sentence_${TOKEN_NUM}_${DATA_NAME}.jsonl"
                OUTPUT_FILE_PATH="${DATA_DIR_PATH}/${MODEL_NAME}/scored/${TRANING_TYPE}-${SCORE_MODEL_TYPE}-${SCORE_MODEL_NAME}/scored_sentence_${TOKEN_NUM}_${DATA_NAME}.jsonl"
                python $LAUNCH_FILE_PATH \
                --src_data_path $SRC_DATA_PATH \
                --output_path $OUTPUT_FILE_PATH \
                --tgt_dim_list $TGT_DIM \
                --score_mode $SCORE_MODE \
                --used_model $SCORE_MODEL_TYPE \
                --used_model_path $SCORE_MODEL_NAME_PATH
            done
        done
    done
done