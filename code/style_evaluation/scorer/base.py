import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class Scorer(object):
    
    def __init__(self, model_name_or_path: str, with_zero = False,is_vllm: bool  = False, **kwargs):
        
        self.is_vllm = is_vllm
        self.with_zero = with_zero
        if not is_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="auto")
        else:
            
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(model_name_or_path)
            self.sampling_params = SamplingParams(max_tokens = 2, logprobs = 1000)
        
    def infer_score(self, user_input: str):

        max_length = 2
        
        if self.is_vllm:
            outputs = self.llm.generate(user_input, self.sampling_params)
            # score_template = np.array([1,2,3,4,5,6])
            # shs modified
            # 目前使用的是[1~5]
            score_template = np.array([1,2,3,4,5])
            if self.with_zero:
                score_template = np.array([0,1,2,3,4,5])
            
            try:
                logprobs_list = outputs[0].outputs[0].logprobs[0]
            except IndexError:
                return 3.0
        else:
            input_ids = self.tokenizer.encode(user_input, return_tensors = "pt").cuda()
            outputs = self.model.generate(input_ids, max_new_tokens = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
            
            try:
                logprobs_list = outputs.scores[0][0].cpu()
            except IndexError:
                return 3.0
            
        score_logits = []
        # score_template = np.array([1,2,3,4,5,6])
        # shs modified
        # 目前使用的是[1~5]
        score_template = np.array([1,2,3,4,5])
        if self.with_zero:
                score_template = np.array([0,1,2,3,4,5])
        for k in self.id2score:
            try:
                score_logits.append(logprobs_list[k])
                # import IPython
                # Ipython.embed()
            except KeyError:
                return 3.0
                
        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)
        score_npy = score_npy * score_template

        score_npy = np.sum(score_npy, axis=0)
        
        return score_npy
            
    def infer_interactivity(self, input_text: str):
        
        interactivity_template = self.interactivity_template
        user_input = interactivity_template.format(sentence=input_text)
        
        return self.infer_score(user_input)

    @property
    def id2score(self):
        raise NotImplementedError
    
    @property
    def interactivity_template(self):
        raise NotImplementedError
    
    def infer_vividness(self, input_text: str):
        
        vividness_template = self.vividness_template
        user_input = vividness_template.format(sentence=input_text)
        
        return self.infer_score(user_input)
    
    @property
    def vividness_template(self):
        raise NotImplementedError
    
    def infer_orality(self, input_text: str):
        
        orality_template = self.orality_template
        user_input = orality_template.format(sentence=input_text)
        
        return self.infer_score(user_input)
    
    @property
    def orality_template(self):
        raise NotImplementedError
    
    def infer_emotionality(self, input_text: str):
        
        emotionality_template = self.emotionality_template
        user_input = emotionality_template.format(sentence=input_text)
        
        return self.infer_score(user_input)
    
    @property
    def emotionality_template(self):
        raise NotImplementedError
    
    # TODO
    # 其他维度的模版和推理代码