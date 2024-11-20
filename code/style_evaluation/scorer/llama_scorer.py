
from .base import Scorer

class Llama2_Scorer(Scorer):
    
    @property
    def id2score(self):
        
        id2score_without_0 = {
                29896: "1",
                29906: "2",
                29941: "3",
                29946: "4",
                29945: "5",
                # 29953: "6"
                }
        id2score_with_0 = {
                29900: "0",
                29896: "1",
                29906: "2",
                29941: "3",
                29946: "4",
                29945: "5",
                }

        if not self.with_zero:
            return id2score_without_0
        else:
            return id2score_with_0
    
    @property
    def interactivity_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the interactivity score of the following sentence. \n##Sentence: {sentence}  \n##Interactivity: ")
        
        return complexity_template
    
    @property
    def vividness_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the vividness score of the following sentence. \n##Sentence: {sentence}  \n##Vividness: ")
        
        return complexity_template
    @property
    def orality_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the orality score of the following sentence. \n##Sentence: {sentence}  \n##Orality: ")
        
        return complexity_template
    @property
    def emotionality_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the emotionality score of the following sentence. \n##Sentence: {sentence}  \n##Emotionality: ")
        
        return complexity_template
    # TODO
    # 其他维度的模版


class Llama3_Scorer(Scorer):
    
    @property
    def id2score(self):
        
        id2score_without_0 = {
                16: "1",
                17: "2",
                18: "3",
                19: "4",
                20: "5",
                # 29953: "6"
                }
        id2score_with_0 = {
                15: "0",
                16: "1",
                17: "2",
                18: "3",
                19: "4",
                20: "5",
                # 29953: "6"
                }
        # 目前只是使用5个分数
        if not self.with_zero:
            return id2score_without_0
        else:
            return id2score_with_0
    
    @property
    def interactivity_template(self):
        
        complexity_template = ("You are a helpful assistant. Please identify the interactivity score of the following sentence. \n##Sentence: {sentence}  \n##Interactivity: ")
        
        return complexity_template
    # TODO
    # 其他维度的模版