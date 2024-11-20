"""
passage 2 sentences
"""
import stanza
import argparse
import os

from functools import partial
from datasets import load_dataset

STANZA_RESOURCES_DIR=os.environ["STANZA_RESOURCES_DIR"]

nlp = stanza.Pipeline(lang='en', processors='tokenize',dir=STANZA_RESOURCES_DIR)

colums_to_segment = ["predictions","src_passage","prediction"]

def sentence_segmentation(doc):
    doc = nlp(doc)
    return [sentence.text for sentence in doc.sentences]

def sentence_segmentation_by_special_symbol(doc):
    """
    split by '\n'
    """
    return doc.split("\n")

def chunk_sentences_by_size(sentences_list,chunk_size):
    result_list = []
    sublist = []
    for sentence in sentences_list:
        sublist.append(sentence)
        if len(sublist) == chunk_size:
            result_list.append(' '.join(sublist))
            sublist.clear()
    if sublist:
        result_list.append(' '.join(sublist))    
    return result_list

def chunk_sentences_by_window_size(sentences_list,chunk_size):
    result_list = []
    num_sentence = len(sentences_list)
    for i in range(num_sentence - chunk_size + 1 ):
        result_list.append(" ".join(sentences_list[i:i+chunk_size]))
    return result_list

def chunk_sentences_by_num(sentences_list,chunk_num):
    chunk_size = len(sentences_list)//chunk_num
    assert chunk_size >= 1
    result_list = chunk_sentences_by_size(sentences_list,chunk_size)
    if len(result_list) != chunk_num:
        result_list[-2] = result_list[-2] + " " + result_list[-1]
        del result_list[-1]
    return result_list

def segment_function_by_chunk_size(example,colums_names,chunk_size_list=[3,5,7]):
    for key in colums_to_segment:
        if key in colums_names:
            sentences = sentence_segmentation(example[key])
            example[f"{key}_sentences_per_1"] = sentences
            for chunk_size in chunk_size_list:
               example[f"{key}_sentences_per_{chunk_size}"] = chunk_sentences_by_size(sentences,chunk_size)
    return example

def segment_function_by_chunk_num(example,colums_names,chunk_num_list=[10,20,30]):
    for key in colums_to_segment:
        if key in colums_names:
            sentences = sentence_segmentation(example[key])
            example[f"{key}_sentences_per_1"] = sentences
            for chunk_num in chunk_num_list:
                example[f"{key}_sentences_with_{chunk_num}"] = chunk_sentences_by_num(sentences,chunk_num)
    return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path",type=str)
    parser.add_argument("--output_path",type=str)
    args = parser.parse_args()
    test_dataset = load_dataset("json",data_files={"train":args.src_data_path},split="train")
    if "prediction" in test_dataset.column_names:
        test_dataset = test_dataset.rename_column("prediction", "predictions")
    colums_names = test_dataset.column_names
    chunk_size_list = [2,3,4]
    process_func = partial(segment_function_by_chunk_size,colums_names=colums_names,chunk_size_list=chunk_size_list)
    test_dataset = test_dataset.map(process_func,num_proc=1)

    os.makedirs(os.path.dirname(args.output_path),exist_ok=True)
    print(test_dataset)
    test_dataset.to_json(args.output_path,force_ascii=False)
    print(f"save results to {args.output_path}")

if __name__ == "__main__":
    main()