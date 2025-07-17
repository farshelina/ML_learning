import nlpaug.augmenter.word as naw
from sklearn.utils import shuffle
from datasets import Dataset

class GLUEAugmenter:
    AUGMENTERS = {
        'synonym': lambda: naw.SynonymAug(aug_src='wordnet', aug_max=3),
        'random_del': lambda: naw.RandomWordAug(action='delete', aug_max=3)
    }

    @classmethod
    def augment_dataset(cls, dataset: Dataset, method: str, multiplier: int = 1):
        """应用数据增强（移除了未实现的回译）"""
        if method not in cls.AUGMENTERS:
            raise ValueError(f"不支持的增强方法: {method}。可选: {list(cls.AUGMENTERS.keys())}")
            
        aug = cls.AUGMENTERS[method]()
        texts = dataset['text'] if 'text' in dataset.features else dataset['sentence1']
        
        # 原始数据 + 增强数据
        augmented = [aug.augment(str(text)) for text in texts * multiplier]  # 确保text是字符串
        full_data = {
            'text': list(texts) + augmented,
            'label': list(dataset['label']) * (multiplier + 1)
        }
        
        return Dataset.from_dict(shuffle(full_data))