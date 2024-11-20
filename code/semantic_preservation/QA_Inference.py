from transformers import pipeline,AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from functools import partial
from tqdm.auto import tqdm
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import re
import argparse
import os
import json


qa_instruction = "Read the article, and answer the question by replying A, B, C or D."

def apply_chat_template(example,tokenizer,model_name_or_path):
    prompt = qa_instruction + "\n\n" + "Article:\n{passage}\n\nQ: {question}".format_map(example) + "\n\n" + "A. {A}\nB. {B}\nC. {C}\nD. {D}".format_map(example["options"])
    messages = [{"role": "user", "content": prompt}]
    prompt_with_temp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    example["prompt"] = prompt_with_temp
    return example

def compute_acc(example):
    if example["extract_ans"] == example["correctAnswer"]:
        example["correct"] = True
    else:
        example["correct"] = False
    return example

def first_option_postprocess(example: str, options: str = "ABCD") -> str:
    """Find first valid option for text."""
    text = example["predictions"]
    # the most accurate answer is:
    patterns = [
        f'[Tt]he answer is [{options}]',
        f'[Tt]he most likely answer is [{options}]',
        f'[Tt]he most accurate answer is [{options}]',
        f'[Tt]he most accurate answer is: [{options}]',
        f'[Tt]he best answer is [{options}]',
        f'[Tt]he correct answer is [{options}]',
        f'[Tt]he answer to the question is [{options}]',
        f'[Tt]he answer to the question is: [{options}]',
        f'[Hh]ere\'s my answer: [{options}]',
        f'Answer: [{options}]',
        f'答案是(.*?)[{options}]',
        f'答案为(.*?)[{options}]',
        f'固选(.*?)[{options}]',
        f'答案应该是(.*?)[{options}]',
        f'(\s|^)[{options}][\s。，,\.$]',  # noqa
        f'[{options}]',
    ]
    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.search(text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    example["extract_ans"] = i
                    example = compute_acc(example)
                    return example
    example["extract_ans"] = ""
    example["correct"] = False
    return example



def find_indexes(lst, target):
    indexes = []
    for i in range(
        len(lst)):
        if lst[i] == target:
            indexes.append(i)
    return indexes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path",type=str)
    parser.add_argument("--output_path",type=str)
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--psst",action="store_true")
    return parser.parse_args()




def main():
    args = get_args()
    print("############"*2,f"PSST: {args.psst}","############"*2)
    # load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "Llama-3-70B-Instruct-GPTQ" in args.model_path:
        print("model is Llama-3-70B-Instruct-GPTQ")
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False
        )
        model = AutoGPTQForCausalLM.from_quantized(
                args.model_path,
                use_safetensors=True,
                device_map="auto",
                quantize_config=quantize_config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        generator = pipeline("text-generation",model=model,tokenizer=tokenizer)
    else:
        generator=pipeline(task="text-generation",model=args.model_path,device_map="auto") 
    

    if generator.tokenizer.pad_token_id is None:
       generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    #    for decoder-only model batching mode
    generator.tokenizer.padding_side = "left"
    # load dataset
    test_dataset=load_dataset("json",data_files={"train":args.src_data_path},split="train")
    # test_dataset=load_dataset("json",data_files={"train":args.src_data_path},split="train").select(range(4))
    # processing
    process_func = partial(apply_chat_template,tokenizer = generator.tokenizer,model_name_or_path = args.model_path)
    test_dataset = test_dataset.map(process_func,num_proc=4)
    infer_dataset=KeyDataset(test_dataset,"prompt")
    # inference
    kwargs={
        "return_full_text":False,
        "max_new_tokens":2048,
        "do_sample":True,
        "top_p":0.95,
        "temperature":1,
        "batch_size":args.batch_size,
    }
    if "Llama-3" in args.model_path:
        terminators = [
        generator.tokenizer.eos_token_id,
        generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        kwargs["eos_token_id"] = terminators

    prediction_list=[]
    for outputs in tqdm(generator(infer_dataset,**kwargs), total=len(test_dataset)):
        predictions=list(map(lambda x:x["generated_text"],outputs))
        prediction_list.extend(predictions)
        
    result_dataset = test_dataset.add_column(name="predictions",column=prediction_list)
    result_dataset = result_dataset.map(first_option_postprocess) 
    os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
    result_dataset.to_json(args.output_path,force_ascii=False)
    print(f"save results to {args.output_path}")
    correct_list = result_dataset["correct"]
    acc = round(sum(correct_list)/len(correct_list)*100,2)
    incorrect_idx = find_indexes(correct_list,False)
    with open(args.output_path.replace(".jsonl","acc.json"), "w") as tgt:
        json.dump({
            "acc": acc,
            "incorrect_idx": incorrect_idx
        },tgt,ensure_ascii=False,indent=4)


if __name__ == "__main__":
    main()