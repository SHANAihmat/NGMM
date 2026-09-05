# import os
# import torch
# import numpy as np
# import pygrinder
# from torch.utils.data import Dataset, DataLoader

# # ==========================================
# # 1. 数据标准化器 (Standard Scaler)
# # ==========================================
# class StandardScaler:
#     def __init__(self, mean=None, std=None):
#         self.mean = mean
#         self.std = std

#     def fit(self, data):
#         self.mean = np.nanmean(data, axis=(0, 1), keepdims=True)
#         self.std = np.nanstd(data, axis=(0, 1), keepdims=True)
#         self.std[self.std == 0] = 1.0

#     def transform(self, data):
#         if self.mean is None or self.std is None:
#             raise ValueError("Scaler has not been fitted yet.")
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean


# # ==========================================
# # 2. ★ 数据增强类（新增）
# # ==========================================
# class TimeSeriesAugmentation:
#     """
#     时间序列数据增强
#     只在训练时使用，验证和测试时不使用
#     """
#     def __init__(
#         self, 
#         noise_std=0.01,           # 高斯噪声标准差
#         scale_range=(0.95, 1.05), # 随机缩放范围
#         shift_range=(-1, 1),      # 时间平移范围
#         dropout_rate=0.1,         # 随机dropout比例
#         enable_noise=True,
#         enable_scale=True,
#         enable_shift=True,
#         enable_dropout=True,
#         augment_prob=0.5          # 每种增强的应用概率
#     ):
#         self.noise_std = noise_std
#         self.scale_range = scale_range
#         self.shift_range = shift_range
#         self.dropout_rate = dropout_rate
#         self.enable_noise = enable_noise
#         self.enable_scale = enable_scale
#         self.enable_shift = enable_shift
#         self.enable_dropout = enable_dropout
#         self.augment_prob = augment_prob
    
#     def add_noise(self, data):
#         """添加高斯噪声"""
#         if not self.enable_noise or np.random.rand() > self.augment_prob:
#             return data
#         noise = np.random.randn(*data.shape) * self.noise_std
#         return data + noise
    
#     def random_scale(self, data):
#         """随机缩放"""
#         if not self.enable_scale or np.random.rand() > self.augment_prob:
#             return data
#         scale = np.random.uniform(*self.scale_range)
#         return data * scale
    
#     def time_shift(self, data):
#         """时间维度平移"""
#         if not self.enable_shift or np.random.rand() > self.augment_prob:
#             return data
#         nodes, seq_len, features = data.shape
#         if seq_len <= 2:
#             return data
#         shift = np.random.randint(*self.shift_range)
#         if shift == 0:
#             return data
#         # 沿时间维度滚动
#         return np.roll(data, shift, axis=1)
    
#     def random_dropout(self, data, mask):
#         """
#         随机dropout一些观测值
#         注意：这里会同时修改 data 和 mask
#         """
#         if not self.enable_dropout or np.random.rand() > self.augment_prob:
#             return data, mask
        
#         # 生成dropout mask
#         dropout_mask = np.random.rand(*data.shape) > self.dropout_rate
#         dropout_mask = dropout_mask.astype(np.float32)
        
#         # 只在原本有观测的地方dropout
#         new_mask = mask * dropout_mask
#         new_data = data * new_mask
        
#         return new_data, new_mask
    
#     def __call__(self, obs_data, era5_data, mask):
#         """
#         应用所有增强
        
#         Args:
#             obs_data: [Nodes, Seq_Len, Features] 观测数据
#             era5_data: [Nodes, Seq_Len, Features] ERA5数据
#             mask: [Nodes, Seq_Len, Features] 缺失mask
        
#         Returns:
#             增强后的 (obs_data, era5_data, mask)
#         """
#         # 1. 添加噪声（对观测数据和ERA5数据都加）
#         obs_data = self.add_noise(obs_data)
#         era5_data = self.add_noise(era5_data)
        
#         # 2. 随机缩放（只对观测数据）
#         obs_data = self.random_scale(obs_data)
        
#         # 3. 时间平移（同时平移观测和ERA5，保持对齐）
#         if np.random.rand() < self.augment_prob:
#             shift = np.random.randint(*self.shift_range)
#             if shift != 0:
#                 obs_data = np.roll(obs_data, shift, axis=1)
#                 era5_data = np.roll(era5_data, shift, axis=1)
#                 mask = np.roll(mask, shift, axis=1)
        
#         # 4. 随机dropout（会修改mask）
#         obs_data, mask = self.random_dropout(obs_data, mask)
        
#         return obs_data, era5_data, mask


# # ==========================================
# # 3. PyTorch Dataset 类（集成数据增强）
# # ==========================================
# class SpatioTemporalDataset(Dataset):
#     def __init__(
#         self, 
#         obs_data, 
#         era5_data, 
#         seq_len, 
#         stride=1, 
#         mode='train', 
#         missing_ratio=0.2, 
#         missing_type='point',
#         augmentation=None  # ★ 新增：数据增强器
#     ):
#         self.obs_data = obs_data
#         self.era5_data = era5_data
#         self.seq_len = seq_len
#         self.stride = stride
#         self.mode = mode
#         self.missing_ratio = missing_ratio
#         self.missing_type = missing_type
#         self.augmentation = augmentation  # ★ 保存增强器
        
#         self.num_nodes, self.total_time, self.num_features = self.obs_data.shape
#         self.num_samples = (self.total_time - self.seq_len) // self.stride + 1

#     def __len__(self): 
#         return self.num_samples

#     def __getitem__(self, idx):
#         start_time = idx * self.stride
#         end_time = start_time + self.seq_len
        
#         obs_window = self.obs_data[:, start_time:end_time, :].copy()
#         era5_window = self.era5_data[:, start_time:end_time, :].copy()
        
#         # 确定缺失比例并设置随机种子
#         if self.mode == 'train':
#             ratio = np.random.uniform(self.missing_ratio * 0.5, self.missing_ratio * 1.5)
#             ratio = min(max(ratio, 0.05), 0.95)
#             np.random.seed(None)  # 训练集保证随机
#         else:
#             ratio = self.missing_ratio
#             np.random.seed(idx + 2026)  # 验证/测试集固定种子
            
#         # ==========================================
#         # 使用 PyGrinder 注入缺失
#         # ==========================================
#         if self.missing_type == 'point_mcar':
#             # # 自己实现的 MCAR
#             # random_matrix = np.random.rand(*obs_window.shape)
#             # mask = (random_matrix >= ratio).astype(np.float32)
#             # masked_obs = obs_window * mask

#             # 使用 PyGrinder 模拟 MCAR (Missing Completely At Random - 完全随机缺失)
#             # PyGrinder 会在缺失位置注入 np.nan
#             X_missing = pygrinder.mcar(obs_window, ratio)
    
#             # 生成 Mask (1 表示未缺失，0 表示缺失)
#             mask = (~np.isnan(X_missing)).astype(np.float32)
    
#             # 将 np.nan 替换为 0.0，以兼容你后续的模型计算逻辑
#             masked_obs = np.nan_to_num(X_missing, nan=0.0)
            
#         elif self.missing_type == 'point_mar':
#             nodes, seq_len, features = obs_window.shape
#             X_2d = obs_window.reshape(-1, features)
#             X_missing_2d = pygrinder.mar_logistic(X_2d, obs_rate=0.3, missing_rate=ratio)
            
#             if isinstance(X_missing_2d, torch.Tensor):
#                 X_missing_2d = X_missing_2d.cpu().numpy()
                
#             X_missing = X_missing_2d.reshape(nodes, seq_len, features)
#             mask = (~np.isnan(X_missing)).astype(np.float32)
#             masked_obs = np.nan_to_num(X_missing, nan=0.0)

#         elif self.missing_type in ['point_mnar_x', 'point_mnar_t', 'point_mnar_nonuniform']:
#             nodes, seq_len, features = obs_window.shape
#             X_2d_time_first = obs_window.transpose(1, 0, 2).reshape(seq_len, nodes * features)
            
#             if self.missing_type == 'point_mnar_x':
#                 X_missing_2d = pygrinder.mnar_x(obs_window, offset=0)
#             elif self.missing_type == 'point_mnar_t':
#                 X_missing_2d = pygrinder.mnar_t(obs_window, cycle=24, pos=10, scale=3)
#             elif self.missing_type == 'point_mnar_nonuniform':
#                 X_missing_2d, _ = pygrinder.mnar_nonuniform(obs_window, p=ratio)

#             if isinstance(X_missing_2d, torch.Tensor):
#                 X_missing_2d = X_missing_2d.cpu().numpy()
                
#             X_missing = X_missing_2d.reshape(seq_len, nodes, features).transpose(1, 0, 2)
#             mask = (~np.isnan(X_missing)).astype(np.float32)
#             masked_obs = np.nan_to_num(X_missing, nan=0.0)

#         # elif self.missing_type == 'seq_mcar':
#         #     nodes, seq_len, features = obs_window.shape
#         #     X_missing = pygrinder.seq_missing(obs_window, p=ratio, seq_len=int(seq_len*0.3))
#         #     mask = (~np.isnan(X_missing)).astype(np.float32)
#         #     masked_obs = np.nan_to_num(X_missing, nan=0.0)

#         elif self.missing_type == 'seq_mcar':
#             nodes, seq_len, features = obs_window.shape
            
#             # 设定序列缺失长度，比如总长的 30%
#             m_seq_len = max(1, int(seq_len * 0.3))
            
#             # 直接把你的目标 ratio 传给 p 即可！
#             X_missing = pygrinder.seq_missing(obs_window, p=ratio, seq_len=m_seq_len)
            
#             mask = (~np.isnan(X_missing)).astype(np.float32)
#             masked_obs = np.nan_to_num(X_missing, nan=0.0)


#         elif self.missing_type in ['seq', 'block_2d']:
#             X_missing = np.empty_like(obs_window)
#             nodes, seq_len, features = obs_window.shape
            
#             if self.missing_type == 'seq':
#                 missing_len = max(1, int(seq_len * 0.3)) 
#                 X_mis_2d = pygrinder.seq_missing(obs_window, p=ratio, seq_len=missing_len)
#             elif self.missing_type == 'block_2d':
#                 b_len = max(1, int(seq_len * 0.3))
#                 b_width = max(1, int(features * 0.5)) 
#                 X_mis_2d = pygrinder.block_missing(obs_window, factor=ratio, block_len=b_len, block_width=b_width)
            
#             if isinstance(X_mis_2d, torch.Tensor):
#                 X_mis_2d = X_mis_2d.cpu().numpy()
                    
#             X_missing = X_mis_2d
#             mask = (~np.isnan(X_missing)).astype(np.float32)
#             masked_obs = np.nan_to_num(X_missing, nan=0.0)


            
#         else:
#             raise ValueError(f"Unsupported missing_type: {self.missing_type}")

#         # 恢复 numpy 全局种子状态
#         if self.mode != 'train':
#             np.random.seed(None)

#         # ==========================================
#         # ★ 应用数据增强（只在训练时）
#         # ==========================================
#         if self.mode == 'train' and self.augmentation is not None:
#             masked_obs, era5_window, mask = self.augmentation(masked_obs, era5_window, mask)

#         return (
#             torch.FloatTensor(masked_obs), 
#             torch.FloatTensor(mask), 
#             torch.FloatTensor(era5_window), 
#             torch.FloatTensor(obs_window)
#         )


# # ==========================================
# # 4. 获取 DataLoader 的主入口函数（添加增强选项）
# # ==========================================
# def get_dataloaders(
#     dataset_dir, 
#     seq_len=48, 
#     train_stride=24, 
#     test_stride=48, 
#     batch_size=32, 
#     train_missing_ratio=0.2, 
#     test_missing_ratio=0.2, 
#     missing_type='point',
#     # ★ 新增：数据增强参数
#     use_augmentation=False,
#     aug_noise_std=0.01,
#     aug_scale_range=(0.95, 1.05),
#     aug_shift_range=(-1, 1),
#     aug_dropout_rate=0.1,
#     aug_prob=0.5
# ):
#     print(f"Loading data from {dataset_dir}...")
#     train_obs = np.load(os.path.join(dataset_dir, 'obs', 'train.npy'))
#     val_obs   = np.load(os.path.join(dataset_dir, 'obs', 'val.npy'))
#     test_obs  = np.load(os.path.join(dataset_dir, 'obs', 'test.npy'))
    
#     train_era5 = np.load(os.path.join(dataset_dir, 'era5', 'train.npy'))
#     val_era5   = np.load(os.path.join(dataset_dir, 'era5', 'val.npy'))
#     test_era5  = np.load(os.path.join(dataset_dir, 'era5', 'test.npy'))
    
#     print("Normalizing data based on training set...")
#     obs_scaler = StandardScaler()
#     obs_scaler.fit(train_obs)
#     train_obs = obs_scaler.transform(train_obs)
#     val_obs   = obs_scaler.transform(val_obs)
#     test_obs  = obs_scaler.transform(test_obs)
    
#     era5_scaler = StandardScaler()
#     era5_scaler.fit(train_era5)
#     train_era5 = era5_scaler.transform(train_era5)
#     val_era5   = era5_scaler.transform(val_era5)
#     test_era5  = era5_scaler.transform(test_era5)
    
#     # ★ 创建数据增强器（只用于训练集）
#     augmentation = None
#     if use_augmentation:
#         augmentation = TimeSeriesAugmentation(
#             noise_std=aug_noise_std,
#             scale_range=aug_scale_range,
#             shift_range=aug_shift_range,
#             dropout_rate=aug_dropout_rate,
#             augment_prob=aug_prob
#         )
#         print(f"✓ Data augmentation enabled for training set")
#         print(f"  - Noise std: {aug_noise_std}")
#         print(f"  - Scale range: {aug_scale_range}")
#         print(f"  - Shift range: {aug_shift_range}")
#         print(f"  - Dropout rate: {aug_dropout_rate}")
#         print(f"  - Augment prob: {aug_prob}")
    
#     train_dataset = SpatioTemporalDataset(
#         train_obs, train_era5, seq_len, stride=train_stride, 
#         mode='train', missing_ratio=train_missing_ratio, missing_type=missing_type,
#         augmentation=augmentation  # ★ 传入增强器
#     )
    
#     val_dataset = SpatioTemporalDataset(
#         val_obs, val_era5, seq_len, stride=test_stride, 
#         mode='val', missing_ratio=test_missing_ratio, missing_type=missing_type,
#         augmentation=None  # 验证集不使用增强
#     )
    
#     test_dataset = SpatioTemporalDataset(
#         test_obs, test_era5, seq_len, stride=test_stride, 
#         mode='test', missing_ratio=test_missing_ratio, missing_type=missing_type,
#         augmentation=None  # 测试集不使用增强
#     )
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     print(f"Dataset created: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}, Test samples={len(test_dataset)}")
    
#     return train_loader, val_loader, test_loader, obs_scaler, era5_scaler
import os
import torch
import numpy as np
import pygrinder
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 数据标准化器 (Standard Scaler)
# ==========================================
class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.nanmean(data, axis=(0, 1), keepdims=True)
        self.std = np.nanstd(data, axis=(0, 1), keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ==========================================
# 2. ★ 数据增强类
# ==========================================
class TimeSeriesAugmentation:
    """
    时间序列数据增强
    只在训练时使用，验证和测试时不使用
    """
    def __init__(
        self, 
        noise_std=0.01,           
        scale_range=(0.95, 1.05), 
        shift_range=(-1, 1),      
        dropout_rate=0.1,         
        enable_noise=True,
        enable_scale=True,
        enable_shift=True,
        enable_dropout=True,
        augment_prob=0.5          
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.dropout_rate = dropout_rate
        self.enable_noise = enable_noise
        self.enable_scale = enable_scale
        self.enable_shift = enable_shift
        self.enable_dropout = enable_dropout
        self.augment_prob = augment_prob
    
    def add_noise(self, data):
        """添加高斯噪声"""
        if not self.enable_noise or np.random.rand() > self.augment_prob:
            return data
        noise = np.random.randn(*data.shape) * self.noise_std
        return data + noise
    
    def random_scale(self, data):
        """随机缩放"""
        if not self.enable_scale or np.random.rand() > self.augment_prob:
            return data
        scale = np.random.uniform(*self.scale_range)
        return data * scale
    
    def time_shift(self, data):
        """时间维度平移"""
        if not self.enable_shift or np.random.rand() > self.augment_prob:
            return data
        nodes, seq_len, features = data.shape
        if seq_len <= 2:
            return data
        shift = np.random.randint(*self.shift_range)
        if shift == 0:
            return data
        return np.roll(data, shift, axis=1)
    
    def random_dropout(self, data, mask):
        """随机dropout一些观测值"""
        if not self.enable_dropout or np.random.rand() > self.augment_prob:
            return data, mask
        
        dropout_mask = np.random.rand(*data.shape) > self.dropout_rate
        dropout_mask = dropout_mask.astype(np.float32)
        
        new_mask = mask * dropout_mask
        new_data = data * new_mask
        
        return new_data, new_mask
    
    def __call__(self, obs_data, era5_data, mask):
        """应用所有增强 (兼容 era5_data 为 None 的情况)"""
        obs_data = self.add_noise(obs_data)
        if era5_data is not None:
            era5_data = self.add_noise(era5_data)
        
        obs_data = self.random_scale(obs_data)
        
        if np.random.rand() < self.augment_prob:
            shift = np.random.randint(*self.shift_range)
            if shift != 0:
                obs_data = np.roll(obs_data, shift, axis=1)
                mask = np.roll(mask, shift, axis=1)
                if era5_data is not None:
                    era5_data = np.roll(era5_data, shift, axis=1)
        
        obs_data, mask = self.random_dropout(obs_data, mask)
        
        return obs_data, era5_data, mask


# ==========================================
# 3. PyTorch Dataset 类
# ==========================================
class SpatioTemporalDataset(Dataset):
    def __init__(
        self, 
        obs_data, 
        era5_data, 
        seq_len, 
        stride=1, 
        mode='train', 
        missing_ratio=0.2, 
        missing_type='point',
        augmentation=None  
    ):
        self.obs_data = obs_data
        self.era5_data = era5_data
        self.has_era5 = (era5_data is not None)  # ★ 标记是否有 ERA5 数据
        
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.missing_type = missing_type
        self.augmentation = augmentation
        
        self.num_nodes, self.total_time, self.num_features = self.obs_data.shape
        self.num_samples = (self.total_time - self.seq_len) // self.stride + 1

    def __len__(self): 
        return self.num_samples

    def __getitem__(self, idx):
        start_time = idx * self.stride
        end_time = start_time + self.seq_len
        
        obs_window = self.obs_data[:, start_time:end_time, :].copy()
        
        # ★ 如果有 ERA5 数据则切片，否则设为 None
        if self.has_era5:
            era5_window = self.era5_data[:, start_time:end_time, :].copy()
        else:
            era5_window = None
        
        if self.mode == 'train':
            ratio = np.random.uniform(self.missing_ratio * 0.5, self.missing_ratio * 1.5)
            ratio = min(max(ratio, 0.05), 0.95)
        else:
            ratio = self.missing_ratio
            # Keep validation and test masks fixed without perturbing the
            # training RNG state used by subsequent samples.
            rng_state = np.random.get_state()
            np.random.seed(idx + 2026)
            
        # ==========================================
        # 使用 PyGrinder 注入缺失
        # ==========================================
        if self.missing_type == 'point_mcar':
            X_missing = pygrinder.mcar(obs_window, ratio)
            mask = (~np.isnan(X_missing)).astype(np.float32)
            masked_obs = np.nan_to_num(X_missing, nan=0.0)
            
        elif self.missing_type == 'point_mar':
            nodes, seq_len, features = obs_window.shape
            X_2d = obs_window.reshape(-1, features)
            X_missing_2d = pygrinder.mar_logistic(X_2d, obs_rate=0.3, missing_rate=ratio)
            
            if isinstance(X_missing_2d, torch.Tensor):
                X_missing_2d = X_missing_2d.cpu().numpy()
                
            X_missing = X_missing_2d.reshape(nodes, seq_len, features)
            mask = (~np.isnan(X_missing)).astype(np.float32)
            masked_obs = np.nan_to_num(X_missing, nan=0.0)

        elif self.missing_type in ['point_mnar_x', 'point_mnar_t', 'point_mnar_nonuniform']:
            nodes, seq_len, features = obs_window.shape
            
            if self.missing_type == 'point_mnar_x':
                X_missing_2d = pygrinder.mnar_x(obs_window, offset=0)
            elif self.missing_type == 'point_mnar_t':
                X_missing_2d = pygrinder.mnar_t(obs_window, cycle=24, pos=10, scale=3)
            elif self.missing_type == 'point_mnar_nonuniform':
                X_missing_2d, _ = pygrinder.mnar_nonuniform(obs_window, p=ratio)

            if isinstance(X_missing_2d, torch.Tensor):
                X_missing_2d = X_missing_2d.cpu().numpy()
                
            X_missing = X_missing_2d.reshape(seq_len, nodes, features).transpose(1, 0, 2)
            mask = (~np.isnan(X_missing)).astype(np.float32)
            masked_obs = np.nan_to_num(X_missing, nan=0.0)

        elif self.missing_type == 'seq_mcar':
            nodes, seq_len, features = obs_window.shape
            m_seq_len = max(1, int(seq_len * 0.3))
            X_missing = pygrinder.seq_missing(obs_window, p=ratio, seq_len=m_seq_len)
            mask = (~np.isnan(X_missing)).astype(np.float32)
            masked_obs = np.nan_to_num(X_missing, nan=0.0)

        elif self.missing_type in ['seq', 'block_2d']:
            nodes, seq_len, features = obs_window.shape
            
            if self.missing_type == 'seq':
                missing_len = max(1, int(seq_len * 0.3)) 
                X_mis_2d = pygrinder.seq_missing(obs_window, p=ratio, seq_len=missing_len)
            elif self.missing_type == 'block_2d':
                b_len = max(1, int(seq_len * 0.3))
                b_width = max(1, int(features * 0.5)) 
                X_mis_2d = pygrinder.block_missing(obs_window, factor=ratio, block_len=b_len, block_width=b_width)
            
            if isinstance(X_mis_2d, torch.Tensor):
                X_mis_2d = X_mis_2d.cpu().numpy()
                    
            X_missing = X_mis_2d
            mask = (~np.isnan(X_missing)).astype(np.float32)
            masked_obs = np.nan_to_num(X_missing, nan=0.0)
            
        else:
            raise ValueError(f"Unsupported missing_type: {self.missing_type}")

        if self.mode != 'train':
            np.random.set_state(rng_state)

        # ==========================================
        # 应用数据增强
        # ==========================================
        if self.mode == 'train' and self.augmentation is not None:
            masked_obs, era5_window, mask = self.augmentation(masked_obs, era5_window, mask)

        # ★ 将 ERA5 封装为 tensor，如果没有则返回空的 tensor 避免 dataloader 报错
        if self.has_era5 and era5_window is not None:
            era5_tensor = torch.FloatTensor(era5_window)
        else:
            era5_tensor = torch.empty(0)  # 使用空 tensor 维持四元组结构

        return (
            torch.FloatTensor(masked_obs), 
            torch.FloatTensor(mask), 
            era5_tensor, 
            torch.FloatTensor(obs_window)
        )


# ==========================================
# 4. 获取 DataLoader 的主入口函数
# ==========================================
def get_dataloaders(
    dataset_dir, 
    seq_len=48, 
    train_stride=24, 
    test_stride=48, 
    batch_size=32, 
    train_missing_ratio=0.2, 
    test_missing_ratio=0.2, 
    missing_type='point',
    use_augmentation=False,
    aug_noise_std=0.01,
    aug_scale_range=(0.95, 1.05),
    aug_shift_range=(-1, 1),
    aug_dropout_rate=0.1,
    aug_prob=0.5
):
    print(f"Loading data from {dataset_dir}...")
    train_obs = np.load(os.path.join(dataset_dir, 'obs', 'train.npy'))
    val_obs   = np.load(os.path.join(dataset_dir, 'obs', 'val.npy'))
    test_obs  = np.load(os.path.join(dataset_dir, 'obs', 'test.npy'))
    
    print("Normalizing OBS data based on training set...")
    obs_scaler = StandardScaler()
    obs_scaler.fit(train_obs)
    train_obs = obs_scaler.transform(train_obs)
    val_obs   = obs_scaler.transform(val_obs)
    test_obs  = obs_scaler.transform(test_obs)
    
    # ★ 检查是否存在 ERA5 数据集
    era5_dir = os.path.join(dataset_dir, 'era5')
    has_era5 = os.path.exists(era5_dir) and os.path.exists(os.path.join(era5_dir, 'train.npy'))
    
    if has_era5:
        print("ERA5 data found. Loading and normalizing...")
        train_era5 = np.load(os.path.join(era5_dir, 'train.npy'))
        val_era5   = np.load(os.path.join(era5_dir, 'val.npy'))
        test_era5  = np.load(os.path.join(era5_dir, 'test.npy'))
        
        era5_scaler = StandardScaler()
        era5_scaler.fit(train_era5)
        train_era5 = era5_scaler.transform(train_era5)
        val_era5   = era5_scaler.transform(val_era5)
        test_era5  = era5_scaler.transform(test_era5)
    else:
        print("⚠️ ERA5 data NOT found. Running without ERA5...")
        train_era5 = val_era5 = test_era5 = None
        era5_scaler = None
    
    # 创建数据增强器（只用于训练集）
    augmentation = None
    if use_augmentation:
        augmentation = TimeSeriesAugmentation(
            noise_std=aug_noise_std,
            scale_range=aug_scale_range,
            shift_range=aug_shift_range,
            dropout_rate=aug_dropout_rate,
            augment_prob=aug_prob
        )
        print(f"✓ Data augmentation enabled for training set")
    
    train_dataset = SpatioTemporalDataset(
        train_obs, train_era5, seq_len, stride=train_stride, 
        mode='train', missing_ratio=train_missing_ratio, missing_type=missing_type,
        augmentation=augmentation  
    )
    
    val_dataset = SpatioTemporalDataset(
        val_obs, val_era5, seq_len, stride=test_stride, 
        mode='val', missing_ratio=test_missing_ratio, missing_type=missing_type,
        augmentation=None  
    )
    
    test_dataset = SpatioTemporalDataset(
        test_obs, test_era5, seq_len, stride=test_stride, 
        mode='test', missing_ratio=test_missing_ratio, missing_type=missing_type,
        augmentation=None  
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset created: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}, Test samples={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, obs_scaler, era5_scaler
