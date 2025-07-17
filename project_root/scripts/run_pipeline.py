import argparse
import pandas as pd
import time
import sys
from datasets import Dataset
from pipelines.data_cleaning import GLUECleaner
from pipelines.feature_engineering import GLUEFeaturizer
from pipelines.augmentation import GLUEAugmenter
from configs.paths import GLUE_PATHS
from pipelines.data_cleaning import MRPCCleaner

def main(task: str, do_augment: bool = False):
    start_time = time.time()
    print(f"[{time.ctime()}] 开始处理任务: {task}")
    
    try:
        if task == 'mrpc':
            cleaner = MRPCCleaner()  # 使用专用类
        # 或者使用通用类：cleaner = GLUECleaner(task)
        else:
            cleaner = GLUECleaner(task)
        # 1. 数据清洗
        cleaner = GLUECleaner(task)
        for split in ['train', 'dev', 'test']:
            cleaner.process_split(split)
    
        # 2. 特征工程
        featurizer = GLUEFeaturizer(task)
        train_df = pd.read_parquet(GLUE_PATHS["processed"] / task / "train.parquet")
        train_df = featurizer.get_handcrafted_features(train_df)
    
        # 3. 数据增强 (仅训练集)
        if do_augment:
            dataset = Dataset.from_pandas(train_df)
            augmented = GLUEAugmenter.augment_dataset(dataset, 'synonym')
            augmented.to_parquet(GLUE_PATHS["augmented"] / task / "train_aug.parquet")
        print(f"[{time.ctime()}] 成功完成 | 耗时: {time.time()-start_time:.1f}s")
    except Exception as e:
        print(f"[{time.ctime()}] 错误发生: {str(e)}", file=sys.stderr)
        sys.exit(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mrpc", 
                       choices=['mrpc', 'sst-2', 'mnli', 'qqp', 'qnli'])
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    print("=== GLUE数据处理管道 ===")
    
    main(args.task, args.augment)