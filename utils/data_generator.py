import torch
from torch.utils.data import Dataset

import pickle
import numpy as np



class MyData(Dataset):
    def __init__(self):
        self.x = None
        self.y = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

    def to_Tensor(self):
        self.x = torch.tensor(self.x, dtype = torch.float32)
        self.y = torch.tensor(self.y, dtype = torch.int64)
        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def save(self, filename):
        """保存Data对象到文件"""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """从文件加载Data对象"""
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

def random_cov_matrix(dim, scale = 1.0):
    """生成随机协方差矩阵"""
    A = np.random.rand(dim, dim) * scale
    cov = np.dot(A, A.T)
    return cov

def generate_data(num_samples_per_class, dim, main_class_ratios, overlap_factor = 0, scale = 1.0):

    data = MyData()

    # 生成前4个主要类别的数据中心，增大中心之间的距离
    base_distance = 15
    adjusted_distance = base_distance * (1 - overlap_factor)
    centers = np.random.rand(4, dim) * adjusted_distance - (adjusted_distance / 2)
    # centers = np.random.rand(4, dim) * 30 - 15

    data.x = []
    data.y = []

    cov1 = random_cov_matrix(dim, scale)
    cov2 = random_cov_matrix(dim, scale * 1.5)

    cov = cov1
    for i, center in enumerate(centers):
        samples = np.random.multivariate_normal(center, cov, num_samples_per_class[i])
        data.x.append(samples)
        data.y.extend([i] * num_samples_per_class[i])

    # 生成第五和第六个类别的数据，位于前四个类别的交界处
    boundary_centers = [
        (centers[0] + centers[1]) / 2,
        (centers[2] + centers[3]) / 2
    ]


    cov = cov1
    for i, center in enumerate(boundary_centers):
        
        samples = np.random.multivariate_normal(center, cov, num_samples_per_class[i + 4])
        data.x.append(samples)
        data.y.extend([i + 4] * num_samples_per_class[i + 4])
        cov = cov2

    # 生成第七和第八个类别的数据，可以在远离前四个类别的任何地方
    distant_centers = [
        np.random.rand(dim) * 100 + 80,
        np.random.rand(dim) * 100 + 50
    ]

    cov = cov1
    for i, center in enumerate(distant_centers):
        cov = random_cov_matrix(dim, scale)
        samples = np.random.multivariate_normal(center, cov, num_samples_per_class[i + 6])
        data.x.append(samples)
        data.y.extend([i + 6] * num_samples_per_class[i + 6])
        cov = cov2

    data.x = np.vstack(data.x)
    data.y = np.array(data.y)

    # 生成train_mask, val_mask, test_mask
    train_indices = []
    val_indices = []

    for i in range(4):
        start_idx = sum(num_samples_per_class[:i])
        end_idx = start_idx + num_samples_per_class[i]
        indices = np.arange(start_idx, end_idx)
        np.random.shuffle(indices)

        train_size = int(main_class_ratios[0] * num_samples_per_class[i])
        val_size = int(main_class_ratios[1] * num_samples_per_class[i])

        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:train_size + val_size])

    data.train_mask = np.zeros(len(data.y), dtype=bool)
    data.val_mask = np.zeros(len(data.y), dtype=bool)
    data.test_mask = np.zeros(len(data.y), dtype=bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[~data.train_mask & ~data.val_mask] = True

    return data