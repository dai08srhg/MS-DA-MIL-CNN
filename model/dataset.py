# pytorchのデータセット
import torch


class DaMilDataset(torch.utils.data.Dataset):
    """
    DAMIL用のpytorchデータセット
    data_root_dir: 画像が置いてあるディレクトリ
    scale: 用いる倍率
    """
    def __init__(self, data_root_dir: data_root_dir, scale):
        self.bags = []
        self.class_labels = []
        self.domain_labels = []
        # ToDo データの保存形式に合わせて実装

        self.data_num = len(self.labels)
        self.domain_unique_num = len(set(self.domain_labels))  # domainの数
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.bags[idx], self.class_labels[idx], self.domain_labels


class MsDaMilDataset(torch.utils.data.Dataset):
    """
    MSDAMIL用のpytorchデータセット
    data_root_dir: 画像が置いてあるディレクトリ
    """
    def __init__(self, data_root_dir: data_root_dir):
        self.scale1_bags = []
        self.scale2_bags = []
        self.class_labels = []
        self.domain_labels = []
        # ToDo データの保存形式に合わせて実装

        self.data_num = len(self.labels)
        self.domain_unique_num = len(set(self.domain_labels))  # domainの数

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.scale1_bags[idx], self.scale2_bags[idx], self.class_labels[idx]