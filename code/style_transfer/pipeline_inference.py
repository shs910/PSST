from transformers import pipeline,AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from functools import partial
from tqdm.auto import tqdm
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import re
import argparse
import os

paraphrase_instruction="Please paraphrase the following passage. Ensure that the core meaning of the original text is preserved and that the style of writing remains consistent. Avoid changing any specific terms or phrases that are central to the understanding or significance of the passage. Aim for clarity and conciseness in your rephrased version, and ensure that the paraphrase maintains the same tone as the original text."

score_instruction=("The above is a output of a formal-to-speech style transfer task, a good speech style needs to have the following characteristics: ",\
                   "1. Emotionality, that is, the speaker has the appropriate emotional expression of the event mentioned. ",\
                    "2. Vividness, that is, the speaker should try to use vivid language to make the speech easier to understand and interesting. ",\
                    "3. Interactivity, that is, the speaker can use questions, appeals and other ways to properly interact with the audience to stimulate the audience's interest. ",\
                    "4. Orality, that is, the content should be optimized for verbal communication, characterized by the use of simpler vocabulary and sentence structures, alongside an increased prevalence of fillers and abbreviations compared to a official text.",\
                    "Now you need to score the output based on the three dimensions, respectively, with a score range from 0 to 100, where 0 means that the output does not have the characteristics of the corresponding dimension and 100 means that the output has the characteristics of the dimension.",\
                    "You need to reply to the results in the following dictionary format: ",\
                    "{'Emotionality': 'Emotionality score', 'Vividness': ' Vividness score ', 'Interactivity': ' Interactive score ', 'Orality': ' Orality score ',}")

def apply_chat_template(example,tokenizer,model_name_or_path,instruction_type="PSST"):
    if  instruction_type == "paraphrase":
        example["paraphrase_instruction"] = paraphrase_instruction
        prompt = "Passage:\n{src_passage}\n\nInstruction:\n{paraphrase_instruction}".format_map(example)
    
    if  instruction_type == "score":
        example["score_instruction"] = "\n".join(score_instruction)
        prompt = "Passage:\n{passage}\n\nInstruction:\n{score_instruction}".format_map(example)

    if instruction_type == "PSST":
        prompt = "Passage:\n{src_passage}\n\nInstruction:\n{instruction}".format_map(example)
    messages = [{"role": "user", "content": prompt}]
    
    prompt_with_temp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    example["prompt"] = prompt_with_temp
    return example

def post_processing(generation_text):
    """
    process the prediction text into a python dict
    """
    generation_text=generation_text.strip("\n")

    if not generation_text.endswith("}"):
        generation_text+="}"
    pattern = r"\{.*?\}"
    result = re.findall(pattern, generation_text,flags=re.DOTALL)
    if len(result)!=0:
        return result[0]
    else:
        return ""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path",type=str)
    parser.add_argument("--output_path",type=str)
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--paraphrase",action="store_true")
    parser.add_argument("--score",action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    instruction_type = "PSST"
    if args.paraphrase:
        instruction_type = "paraphrase"
    if args.score:
        instruction_type = "score"
    print("############"*2,f"instruction_type: {instruction_type}","############"*2)
    # laod model and tokenizer
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
    # dataset
    test_dataset=load_dataset("json",data_files={"train":args.src_data_path},split="train")
    # test_dataset=load_dataset("json",data_files={"train":args.src_data_path},split="train").select(range(4))
    # pre-processing
    process_func = partial(apply_chat_template,tokenizer = generator.tokenizer,model_name_or_path = args.model_path,instruction_type = instruction_type)
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
    extract_text_list = []
    for outputs in tqdm(generator(infer_dataset,**kwargs), total=len(test_dataset)):
        predictions=list(map(lambda x:x["generated_text"],outputs))
        prediction_list.extend(predictions)
        if args.score:
            extract_text_list.extend([post_processing(t) for t in predictions])

    result_dataset=test_dataset.add_column(name="predictions",column=prediction_list)
    if args.score:
        result_dataset=result_dataset.add_column(name="score",column=extract_text_list)
    os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
    result_dataset.to_json(args.output_path,force_ascii=False)
    print(f"save results to {args.output_path}")


if __name__ == "__main__":
    main()