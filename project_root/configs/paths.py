from pathlib import Path

# 基础路径
PROJECT_ROOT = Path(__file__).parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

# GLUE相关路径
GLUE_PATHS = {
    "raw": DATA_ROOT / "glue" / "raw",
    "processed": DATA_ROOT / "glue" / "processed",
    "augmented": DATA_ROOT / "glue" / "augmented",
    "models": DATA_ROOT / "models"
}

# 创建目录
for path in GLUE_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)