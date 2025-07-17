import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer

class GLUEFeaturizer:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_handcrafted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """传统特征工程"""
        # 句长特征
        df['sentence1_len'] = df['sentence1'].apply(len)
        
        # 词重叠率 (句对任务)
        if 'sentence2' in df.columns:
            df['jaccard_sim'] = df.apply(
                lambda x: len(set(x['sentence1'].split()) & set(x['sentence2'].split())) / 
                        len(set(x['sentence1'].split()) | set(x['sentence2'].split())),
                axis=1
            )
        return df

    def get_bert_embeddings(self, texts: list) -> np.ndarray:
        """生成BERT嵌入"""
        inputs = self.tokenizer(texts, return_tensors="pt", 
                              padding=True, truncation=True, max_length=128)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()  # CLS向量

    def get_semantic_similarity(self, pairs: list) -> np.ndarray:
        """句对语义相似度"""
        embs = self.st_model.encode([p for pair in pairs for p in pair])
        embs = embs.reshape(len(pairs), 2, -1)
        return np.array([np.dot(e[0], e[1])/(np.linalg.norm(e[0])*np.linalg.norm(e[1])) 
        for e in embs])