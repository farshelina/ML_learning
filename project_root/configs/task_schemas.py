"""
定义各GLUE任务的数据结构
"""
TASK_SCHEMAS = {
    'mrpc': {
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
        'max_length': 256  # 每个句子的最大token数
    },
    'sst-2': {
        'text_columns': ['text'],
        'label_column': 'label',
        'max_length': 512
    }
}