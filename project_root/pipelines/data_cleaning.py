import re
import pandas as pd
from pathlib import Path
from configs.paths import GLUE_PATHS
from transformers import BertTokenizer
from typing import List, Union
from configs.task_schemas import TASK_SCHEMAS

class GLUECleaner:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # 添加文本列映射字典
        self.text_columns = {
            'mrpc': ['sentence1', 'sentence2'],
            'sst-2': ['text'],
            'qqp': ['question1', 'question2']
        }
        
    def clean_text(self, text: str, keep_emojis: bool = False) -> str:
        """标准化文本处理"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 特殊字符处理
        if keep_emojis:
            text = re.sub(r'([^\w\s,.!?😊😢])', '', text)  # 保留基本标点和表情
        else:
            text = re.sub(r'([^\w\s,.!?])', '', text)
            
        return text.lower().strip()
    
    def _validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """标签验证"""
        valid_labels = {'mrpc': [0, 1], 'sst-2': [0, 1]}  # 各任务有效标签
        return df[df['label'].isin(valid_labels.get(self.task_name, []))]

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
    #"""通用过滤方法（处理所有GLUE任务）"""
    # 获取当前任务的文本列
        columns = self.text_columns.get(self.task_name, ['text'])
    
    # 检查所有必要的列是否存在
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"数据集缺少必要列: {col} (任务: {self.task_name})")
    
    # 多列长度过滤（如MRPC需要同时检查两个句子）
        mask = pd.Series(True, index=df.index)
        for col in columns:
            mask &= df[col].apply(
                lambda x: len(self.tokenizer.tokenize(str(x))) <= 512
            )
    
        return df[mask]

    def process_split(self, split: str):
        """处理单个split (train/dev/test)"""
        raw_path = GLUE_PATHS["raw"] / self.task_name / f"{split}.csv"
        df = pd.read_csv(raw_path)
        
        # 应用清洗
        text_cols = ['sentence1', 'sentence2'] if 'sentence2' in df.columns else ['text']
        for col in text_cols:
            df[col] = df[col].apply(
                lambda x: self.clean_text(str(x), keep_emojis=self.task_name=='sst-2'))
                
        df = self.filter_data(df)
        
        # 保存处理后的数据
        save_path = GLUE_PATHS["processed"] / self.task_name 
        save_path.mkdir(exist_ok=True)
        df.to_parquet(save_path / f"{split}.parquet")

        print(f"清洗报告 - {split}:")
        print(f"  删除无效标签: {len(df) - len(self.filter_data(df))} 条")
        print(f"  文本标准化示例: {df.iloc[0]['sentence1'][:50]}...")


class MRPCCleaner(GLUECleaner):
    #"""MRPC任务专用清洗器"""
    def __init__(self):
        super().__init__('mrpc')
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 专门处理MRPC的双句长度
        cond1 = df['sentence1'].apply(
            lambda x: len(self.tokenizer.tokenize(str(x))) <= 256
        )
        cond2 = df['sentence2'].apply(
            lambda x: len(self.tokenizer.tokenize(str(x))) <= 256
        )
        return df[cond1 & cond2]