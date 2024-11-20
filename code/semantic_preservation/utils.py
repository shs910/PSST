import json
import random

def write_list2jsonl(obj_list,tgt_path):
    with open(tgt_path,"w") as tgt:
        for item in obj_list:
            tgt.write(json.dumps(item, ensure_ascii=False)+'\n')

def shuffle_choices(qa):
    # print(qa)
    options = qa["options"]
    correctAnswer = qa["correctAnswer"]
    options = list(options.items())
    random.shuffle(options)
    new_options = {}
    for new_k,old_kv in zip(["A","B","C","D"],options):
        new_options[new_k] = old_kv[1]
        if old_kv[0] == correctAnswer:
            qa["correctAnswer"] = new_k
    qa["options"] = new_options
    return qa

def process_data(src_path,tgt_path):
    ttt = "/home/yzyang/LLM/shs/code/PSST/semantic_preservation/qa_based/utils.json"
    with open(src_path,"r") as src,open(ttt,'r') as tt:
        iiiiii = json.load(tt)
        with open(tgt_path,"w") as tgt:
            datas = json.load(src)
            qa_1_list = []
            qa_2_list = []
            for idx,item in enumerate(datas):
                # print(idx)
                tmp_data = {}
                tmp_data["passage"] = item["src_passage"]
                try:
                # tmp_data["QA1_list"] = json.loads(item["qa1"])["questions"]
                # tmp_data["QA2_list"] = json.loads(item["qa2"])["questions"]
                    tmp_data["QA1_list"] = eval(item["qa1"])["questions"]
                    tmp_data["QA2_list"] = eval(item["qa2"])["questions"]
                except:
                    for bbbb in iiiiii:
                        if bbbb["idx"] == idx:
                            tmp_data["QA1_list"] = bbbb["QA1_list"]["questions"]
                            tmp_data["QA2_list"] = bbbb["QA2_list"]["questions"]
                            break
                tgt.write(json.dumps(tmp_data,ensure_ascii=False)+'\n')
                for ii,qa in enumerate(tmp_data["QA1_list"]):
                    
                    qa = shuffle_choices(qa)
                    qa_1_list.append({"idx":idx,
                                      "passage": item["src_passage"],
                                      "q_idx": ii,
                                      "question": qa["question"],
                                      "options": qa["options"],
                                      "correctAnswer":qa["correctAnswer"]
                                      })
                for ii,qa in enumerate(tmp_data["QA2_list"]):
                    qa = shuffle_choices(qa)
                    qa_2_list.append({"idx":idx,
                                      "passage": item["src_passage"],
                                      "q_idx": ii,
                                      "question": qa["question"],
                                      "options": qa["options"],
                                      "correctAnswer":qa["correctAnswer"]
                                      })
            write_list2jsonl(qa_1_list,tgt_path.replace(".jsonl", "_QA1.jsonl"))
            write_list2jsonl(qa_2_list,tgt_path.replace(".jsonl", "_QA2.jsonl"))

if __name__ == "__main__":
    base_path = '/home/yzyang/LLM/shs/code/PSST'
    question_name = "800_sample"
    src_path = f"{base_path}/oral_exp/1_different_level/token_num/raw/800_delta_200/qa/{question_name}.json"
    tgt_path = f"{base_path}/semantic_preservation/qa_based/qa_pairs/{question_name}.jsonl"
    process_data(src_path,tgt_path)