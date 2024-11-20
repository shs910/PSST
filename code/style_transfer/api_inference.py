from openai_access import call_chatgpt
from datasets import load_dataset
from functools import partial
from tqdm import tqdm

import json
import os
import argparse

paraphrase_instruction="Please paraphrase the following text while ensuring the semantics and style of the original are preserved."

def apply_chat_template(example,paraphrase=False):
    if  paraphrase:
        example["paraphrase_instruction"] = paraphrase_instruction
        prompt = "Instruction:\n{paraphrase_instruction}\n\nPassage:\n{src_passage}".format_map(example)
    else:
        prompt = "Passage:\n{src_passage}\n\nInstruction:\n{instruction}".format_map(example)
    example["prompt"] = prompt
    return example

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path",type=str)
    parser.add_argument("--output_path",type=str)
    parser.add_argument("--model_name",type=str,default="gpt-3.5")
    # "gpt-4","gpt-3.5"
    parser.add_argument("--paraphrase",action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    print("############"*2,f"paraphrase? {args.paraphrase}","############"*2)
    test_dataset=load_dataset("json",data_files={"train":args.src_data_path},split="train")
    # test_dataset=load_dataset("json",data_files={"train":args.src_data_path},split="train").select(range(2))
    process_func = partial(apply_chat_template,paraphrase=args.paraphrase)
    test_dataset = test_dataset.map(process_func,num_proc=4)

    os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
    with open(args.output_path,"w") as out:
        for example in tqdm(test_dataset):
            prompt = example["prompt"]
            return_text = call_chatgpt(prompt,args.model_name)
            example["predictions"] = return_text
            out.write(json.dumps(example,ensure_ascii=False)+"\n")
    print(f"save results to {args.output_path}")

if __name__ == "__main__" :
    main()
    