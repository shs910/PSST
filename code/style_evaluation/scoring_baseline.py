from scorer import Llama3_Scorer,Llama2_Scorer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
import json
import argparse

DIM_LIST=["interactivity","vividness","orality","emotionality"]
PASSAGE_TYPE=["src_passage","predictions"]
SCORE_MODE_LIST=["per_1","per_2","per_3","per_4"]
CHUNK_NUM_LIST=[5,10,15,20]

def score_sample(sample,scorer,dim="interactivity",score_mode="per_1",passage_type="predictions"):
    scores_list = []
    sentence_list = sample[f"{passage_type}_sentences_{score_mode}"]
    for sentence  in sentence_list:
        if sentence == "":
            scores_list.append(0)
            continue
        if dim == "interactivity":
            s = scorer.infer_interactivity(sentence)
        elif dim == "vividness":
            s = scorer.infer_vividness(sentence)
        elif dim == "orality":
            s = scorer.infer_orality(sentence)
        elif dim == "emotionality":
            s = scorer.infer_emotionality(sentence)
        else:
            raise KeyError("evaluation dim wrong")
        scores_list.append(s)
    return scores_list

def analyse_scores(example):
    keys = list(example.keys()) 
    for p_t in PASSAGE_TYPE:
        for dim in DIM_LIST:
            for mode in SCORE_MODE_LIST:
                k = f"{dim}_{p_t}_{mode}"
                if k in keys:
                    res = {
                        "mean_score":np.mean(example[k]),
                        "std_score":np.std(example[k]),
                        "max_score":np.max(example[k]),
                        "max_idx":np.argmax(example[k]),
                        "min_score":np.min(example[k]),
                        "min_idx":np.argmin(example[k]),
                    } 
                    example[f"{k}_analysis"] = res
    return example

def merge_scores(scores_distribution_list,chunk_num):
    
    new_scores_distribution_list = []
    short_num = 0
    valid_num = 0
    for scores_distribution in scores_distribution_list:
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
            valid_num+=1
        else:
            short_num+=1
            continue
        assert len(new_scores_distribution)==chunk_num
        new_scores_distribution_list.append(new_scores_distribution)
    return new_scores_distribution_list,short_num,valid_num

def comput_mean_distribution(datasets):
    keys = datasets.column_names
    res_dict = {}
    for p_t in PASSAGE_TYPE:
        for dim in DIM_LIST:
            total_mean_score = []
            for mode in SCORE_MODE_LIST:
                k = f"{dim}_{p_t}_{mode}"
                tmp_res = {}
                if k in keys:
                    scores_distribution_list = datasets[k]
                    for chunk_num in CHUNK_NUM_LIST:
                        try:
                            new_scores_distribution_list,shorter_num,valid_num = merge_scores(scores_distribution_list,chunk_num)
                            scores_distribution = np.array(new_scores_distribution_list)
                            mean_distribution = np.mean(scores_distribution, axis=0).tolist()
                            tmp_res[f"mean_distribution_with_chunk_num_{chunk_num}"] = [round(x*20,2) for x in mean_distribution]                         
                            tmp_res[f"valid_num_with_chunk_num_{chunk_num}"] = valid_num
                            tmp_res[f"shorter_num_with_chunk_num_{chunk_num}"] = shorter_num
                        except:
                            continue
                    empty_num = 0
                    valid_mean_scores = []
                    for old_score_list in scores_distribution_list:
                        if len(old_score_list)!=0:
                            valid_mean_scores.append(np.mean(old_score_list))
                        else:
                            empty_num+=1
                    tmp_res[f"mean_score"] = round(np.mean(valid_mean_scores)*20,2)
                    tmp_res[f"valid_num"] = len(valid_mean_scores)
                    tmp_res[f"empty_num"] = empty_num
                    total_mean_score.append(tmp_res[f"mean_score"])
                res_dict[k] = tmp_res
            if len(total_mean_score) != 0:
                res_dict[f"{dim}_{p_t}_total_mean_score"] = {
                    "diff_model_mean_score_list":total_mean_score,
                    "mean_score":round(np.mean(total_mean_score),2),
                }
    return res_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path",type=str)
    parser.add_argument("--output_path",type=str)
    parser.add_argument("--tgt_dim_list",type=str,default="interactivity")
    parser.add_argument("--score_mode",type=str,default="per_1")
    parser.add_argument("--used_model",type=str,default="llama3")
    parser.add_argument("--with_zero",action="store_true")
    parser.add_argument("--used_model_path",type=str,default=None)
    parser.add_argument("--analysis_only",action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    test_dataset = load_dataset("json",data_files={"train":args.src_data_path},split="train")
    if "prediction" in test_dataset.column_names:
        test_dataset = test_dataset.rename_column("prediction", "predictions")
    if not args.analysis_only:
        if args.used_model == "llama3":
            scorer = Llama3_Scorer(args.used_model_path,with_zero=args.with_zero)
        elif args.used_model == "llama2":
            scorer = Llama2_Scorer(args.used_model_path,with_zero=args.with_zero)
        passage_type_list=["src_passage"]
        for passage_type in passage_type_list:
            print(f"evaluation {passage_type}")
            for tgt_dim in args.tgt_dim_list.split(","):
                print(f"evaluation {passage_type} on {tgt_dim}")
                for score_mode in args.score_mode.split(","):
                    print(f"evaluation {passage_type} on {tgt_dim} with {score_mode}")
                    all_sentence_scores = []
                    for sample in tqdm(test_dataset):
                        tmp_scores_list = score_sample(sample,scorer,dim=tgt_dim,score_mode=score_mode,passage_type=passage_type)
                        all_sentence_scores.append(tmp_scores_list)
                    # add colums
                    test_dataset = test_dataset.add_column(f"{tgt_dim}_{passage_type}_{score_mode}",all_sentence_scores)
        os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
        test_dataset.to_json(args.output_path,force_ascii=False)
        print(test_dataset)
        try:
            test_dataset = test_dataset.map(analyse_scores,num_proc=4)
            print(test_dataset)
            test_dataset.to_json(args.output_path,force_ascii=False)
        except:
            pass
        
        print(f"save results to {args.output_path}")
        res_dict = comput_mean_distribution(test_dataset)
        with open(args.output_path.replace(".jsonl","_means_res.json"),"w") as f:
            json.dump(res_dict,f,ensure_ascii=False,indent=4) 
    else:
        res_dict = comput_mean_distribution(test_dataset)
        with open(args.src_data_path.replace(".jsonl","_means_res.json"),"w") as f:
            json.dump(res_dict,f,ensure_ascii=False,indent=4) 
    
if __name__ == "__main__":
    main()