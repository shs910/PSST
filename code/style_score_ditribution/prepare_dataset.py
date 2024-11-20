
import numpy as np
import os
 
from functools import partial
from datasets import load_dataset,Dataset

BASE_PATH="./code"

DIM_LIST=["interactivity","vividness","emotionality","orality"]
PASSAGE_TYPE=["src_passage","predictions"]
SCORE_MODE_LIST=["per_1","per_2","per_3","per_4"]
CHUNK_NUM_LIST=[5,7,10,15,20]
CHUNK_NUM=7

tgt_dir_path=f"{BASE_PATH}/style_score_ditribution/sample_for_plot/MODE_{len(SCORE_MODE_LIST)}_CHUNK_{CHUNK_NUM}"

def chunk_distribution(scores_distribution,chunk_num):
    sample_nums = len(scores_distribution)
    chunk_size = sample_nums//chunk_num 
    remainder = sample_nums % chunk_num
    new_scores_distribution = []
    if chunk_size >= 1 :
        start = 0
        for i in range(chunk_num):
            chunk_end = start + chunk_size + (1 if i < remainder else 0)
            chunk = scores_distribution[start:chunk_end]
            new_scores_distribution.append(np.mean(chunk))
            start = chunk_end
        return new_scores_distribution,True
    else:
        count=0
        mid = len(scores_distribution)//2
        try:
            mean_value = np.mean(scores_distribution)
            min_value = np.min(scores_distribution)
        except:
            mean_value = 0
            min_value = 0
        while True:
            scores_distribution.insert(mid+count,mean_value)
            count+=1
            if len(scores_distribution) == chunk_num:
                break
    return scores_distribution,False

def process_single_sample(example,dim):
    keys = list(example.keys()) 
    for p_t in PASSAGE_TYPE:
        new_score_distribution_list = []
        all_valid = True
        for mode in SCORE_MODE_LIST:
            k = f"{dim}_{p_t}_{mode}"
            if k in keys:
                scores_distribution = example[k]
                new_scores_distribution,all_valid = chunk_distribution(scores_distribution,CHUNK_NUM)
                new_score_distribution_list.append(new_scores_distribution)
        if len(new_score_distribution_list) != 0:
            example[f"{p_t}_{dim}_socre_distribution"] = np.mean(np.array(new_score_distribution_list),axis=0).tolist()
            example[f"{p_t}_{dim}_socre_distribution_valid"] = all_valid
    return example

def get_plot_sample(dataset:Dataset,passage_type = "src_text",valid_only=True ,model_name = None):
    all_data_dict_list = []
    keys = dataset.column_names
    if "paraphrase" in passage_type:
        passage_type = "paraphrase"
    if "enhanced" in passage_type:
        passage_type = "enhanced"
    if "concise" in passage_type:
        passage_type = "concise"
    model_name_mapping = {"Llama-2-7b-chat-ms": "Llama-2-7b-chat",
              "Llama-2-13b-chat-ms": "Llama-2-13b-chat",
              "Llama-2-70B-Chat-GPTQ": "Llama-2-70b-chat",
              "Llama-3-8B-Instruct": "Llama-3-8b-instruct",
              "Llama-3-70B-Instruct-GPTQ": "Llama-3-70b-instruct",
              "gpt-3.5": "GPT-3.5-turbo",
              }
    for i in range(dataset.num_rows):
        item = dataset[i]
        for p_t in PASSAGE_TYPE:
            for dim in DIM_LIST:
                k = f"{p_t}_{dim}_socre_distribution"
                if k in keys:
                    if valid_only and not item[f"{p_t}_{dim}_socre_distribution_valid"]:
                        continue
                    socre_distribution = item[k]
                    for idx,s in enumerate(socre_distribution):
                        tmp_dict = {
                            "id":f"{i}-{idx}",
                            "text_type":passage_type,
                            "position":idx+1,
                            "score":s*20,
                            "dimension":dim,
                        }
                        if model_name:
                            tmp_dict["model_name"] = model_name_mapping.get(model_name,model_name)
                        all_data_dict_list.append(tmp_dict)
    return all_data_dict_list

def create_baseline_plot_data():
    for valid_only in [True,False]:
        data_type = "token_num"
        dim_list = ["interactivity","vividness","emotionality","orality"]
        src_dir_path = f"{BASE_PATH}/oral_exp/1_different_level/{data_type}/raw"
        for sentence_num in ["1200_delta_200"]:
            print(f"####{sentence_num}####")
            all_sample_dict_list = []
            for dim in dim_list:
                score_model_type = f"{dim}_gpt-3.5_with_out_0-llama2-llama_1.1b"
                for baseline_name in ["ted_text", "src_text", "gpt-3.5_paraphrase"]:
                    print(baseline_name)
                    tmp_data_path = os.path.join(src_dir_path,sentence_num,\
                                    "scored",score_model_type,f"scored_{baseline_name}.jsonl")
                    tmp_dataset = load_dataset("json",data_files={"train":tmp_data_path},split="train")
                    tmp_dataset = tmp_dataset.map(partial(process_single_sample,dim=dim))
                    tmp_data_dict_list = get_plot_sample(tmp_dataset,baseline_name,valid_only)
                    all_sample_dict_list.extend(tmp_data_dict_list)
            all_type_daatset = Dataset.from_list(all_sample_dict_list)
            os.makedirs(os.path.join(tgt_dir_path,"baselines"), exist_ok=True)
            print(all_type_daatset)
            if valid_only:
                all_type_daatset.to_csv(os.path.join(tgt_dir_path,"baselines",f"{sentence_num}_baselines_valid_only.csv"))
            else:
                all_type_daatset.to_csv(os.path.join(tgt_dir_path,"baselines",f"{sentence_num}_baselines.csv"))
            print("###done###")

def create_model_ployt_data():
    data_type = "token_num"
    data_name_list = ["1200_delta_200"]
    
    model_name_list = ["Llama-2-7b-chat-ms", "Llama-2-13b-chat-ms", "Llama-2-70B-Chat-GPTQ", "Llama-3-8B-Instruct", "Llama-3-70B-Instruct-GPTQ","gpt-3.5"]
    dim_list = ["interactivity","vividness","emotionality","orality"]
    prompt_type_list = ["with_concise_prompt", "with_enhanced_prompt_v3"]
    for valid_only in [False,True]:
        src_dir_path = f"{BASE_PATH}/oral_exp/1_different_level/{data_type}/test_res"
        for sentence_num in data_name_list:
            print(f"####{sentence_num}####")
            for model_name in model_name_list:
                try:
                    all_sample_dict_list = []
                    for dim in dim_list:
                        score_model_type = f"{dim}_gpt-3.5_with_out_0-llama2-llama_1.1b"
                        for prompt_type in prompt_type_list:
                            print(model_name,"######",prompt_type,"######",dim)
                            tmp_data_path = os.path.join(src_dir_path,model_name,\
                                            "scored",score_model_type,f"scored_sentence_{sentence_num}_{prompt_type}.jsonl")
                            tmp_dataset = load_dataset("json",data_files={"train":tmp_data_path},split="train")
                            tmp_dataset = tmp_dataset.map(partial(process_single_sample,dim=dim))
                            tmp_data_dict_list = get_plot_sample(tmp_dataset,prompt_type,valid_only,model_name)
                            all_sample_dict_list.extend(tmp_data_dict_list)
                    all_type_daatset = Dataset.from_list(all_sample_dict_list)
                    print(all_type_daatset)
                    os.makedirs(os.path.join(tgt_dir_path,model_name), exist_ok=True)
                    if valid_only:
                        all_type_daatset.to_csv(os.path.join(tgt_dir_path,model_name,f"{sentence_num}_{model_name}_valid_only.csv"))
                    else:
                        all_type_daatset.to_csv(os.path.join(tgt_dir_path,model_name,f"{sentence_num}_{model_name}.csv"))
                    print("###done###")
                except:
                    continue

if __name__ == "__main__":
    create_baseline_plot_data()
    create_model_ployt_data()