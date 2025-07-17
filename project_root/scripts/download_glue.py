import argparse
from datasets import load_dataset
from pathlib import Path
import sys
from configs.paths import GLUE_PATHS

def download_glue(task: str):
    try:
        print("正在下载数据集...")
        dataset = load_dataset('glue', task)
        
        save_dir = GLUE_PATHS["raw"] / task
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n保存目录：{save_dir.absolute()}\n")

        splits = {'train': 'train', 'validation': 'dev', 'test': 'test'}
        for hf_split, local_split in splits.items():
            output_path = save_dir / f"{local_split}.csv"
            print(f"正在处理 {hf_split}...", end=" ")
            
            dataset[hf_split].to_csv(output_path)
            
            if not output_path.exists():
                print("[失败]")
                raise RuntimeError(f"文件未生成：{output_path}")
            print(f"[成功] 大小：{output_path.stat().st_size/1024:.1f} KB")
            
        print("\n所有文件已保存！")
        return True
        
    except Exception as e:
        print(f"\n错误发生：{str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, 
                       choices=['mrpc', 'sst2', 'mnli', 'qqp', 'qnli'])
    args = parser.parse_args()
    
    success = download_glue(args.task)
    sys.exit(0 if success else 1)