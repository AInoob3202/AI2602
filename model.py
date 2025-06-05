# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
from matplotlib import font_manager
warnings.filterwarnings('ignore')
myfont = font_manager.FontProperties(fname='C:\\Windows\\Fonts\\msyh.ttc')
# 设置中文字体支持

sns.set_style("whitegrid")

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理类
class WeatherDataProcessor:
    def __init__(self):
        self.scalers = {}
        self.string_mappings = {}  # 存储字符串到数字的映射
        
    def load_and_preprocess_data(self, file_path):
        """加载和预处理气象数据 - 针对特定数据格式重新设计"""
        print("加载数据...")
        
        df = pd.read_csv(file_path)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 创建副本
        processed_df = df.copy()
        
        # 1. 处理日期列 - 针对 "Formatted Date" 列，手动转换为UTC时间
        date_col = 'Formatted Date'
        if date_col in processed_df.columns:
            print(f"处理日期列: {date_col}")
            
            try:
                # 手动处理时区转换
                processed_df['datetime'] = processed_df[date_col].apply(self._convert_to_utc)
                
                # 检查转换结果
                valid_dates = processed_df['datetime'].notna().sum()
                total_dates = len(processed_df)
                
                print(f"成功转换日期: {valid_dates}/{total_dates} ({valid_dates/total_dates*100:.1f}%)")
                
                if valid_dates > 0:
                    # 打印转换示例
                    print("时间转换示例:")
                    for i in range(min(3, len(processed_df))):
                        if pd.notna(processed_df.iloc[i]['datetime']):
                            original = processed_df.iloc[i][date_col]
                            converted = processed_df.iloc[i]['datetime']
                            print(f"  原始: {original} -> UTC: {converted}")
                
                    # 提取时间特征（基于UTC时间）
                    processed_df['Hour'] = processed_df['datetime'].dt.hour
                    processed_df['Month'] = processed_df['datetime'].dt.month
                    processed_df['Day'] = processed_df['datetime'].dt.day
                    processed_df['DayOfWeek'] = processed_df['datetime'].dt.dayofweek
                    processed_df['DayOfYear'] = processed_df['datetime'].dt.dayofyear
                    
                    # 季节特征 (0:春, 1:夏, 2:秋, 3:冬) - 基于UTC时间
                    processed_df['Season'] = processed_df['Month'].map({
                        12: 3, 1: 3, 2: 3,  # 冬季
                        3: 0, 4: 0, 5: 0,   # 春季
                        6: 1, 7: 1, 8: 1,   # 夏季
                        9: 2, 10: 2, 11: 2  # 秋季
                    })
                    
                    # 时间周期特征 (正弦/余弦编码) - 基于UTC时间
                    processed_df['Hour_sin'] = np.sin(2 * np.pi * processed_df['Hour'] / 24)
                    processed_df['Hour_cos'] = np.cos(2 * np.pi * processed_df['Hour'] / 24)
                    processed_df['Month_sin'] = np.sin(2 * np.pi * processed_df['Month'] / 12)
                    processed_df['Month_cos'] = np.cos(2 * np.pi * processed_df['Month'] / 12)
                    processed_df['DayOfWeek_sin'] = np.sin(2 * np.pi * processed_df['DayOfWeek'] / 7)
                    processed_df['DayOfWeek_cos'] = np.cos(2 * np.pi * processed_df['DayOfWeek'] / 7)
                    
                    print("基于UTC时间的特征提取成功")
                    
                else:
                    print("警告: 无法解析日期，使用默认时间特征")
                    self._create_default_time_features(processed_df)
                    
            except Exception as e:
                print(f"日期处理错误: {e}")
                print("使用默认时间特征")
                self._create_default_time_features(processed_df)
        else:
            print(f"未找到日期列 '{date_col}'，使用默认时间特征")
            self._create_default_time_features(processed_df)
        
        # 2. 处理字符串列 - 对指定的字符串列进行编码
        string_columns = ['Summary', 'Precip Type', 'Daily Summary']
        
        for col in string_columns:
            if col in processed_df.columns:
                print(f"编码字符串列: {col}")
                
                # 填充缺失值
                processed_df[col] = processed_df[col].fillna('unknown')
                
                # 创建唯一值映射
                unique_values = sorted(processed_df[col].unique())
                mapping = {value: idx for idx, value in enumerate(unique_values)}
                
                # 应用编码
                processed_df[f'{col}_encoded'] = processed_df[col].map(mapping)
                
                # 保存映射关系
                self.string_mappings[col] = mapping
                
                print(f"  {col}: {len(unique_values)} 个唯一值")
                print(f"  示例映射: {dict(list(mapping.items())[:3])}")

        # 3. 选择数值特征列
        # 原始数值列
        numeric_columns = [
            'Temperature (C)',
            'Apparent Temperature (C)', 
            'Humidity',
            'Wind Speed (km/h)',
            'Wind Bearing (degrees)',
            'Visibility (km)',
            'Loud Cover',
            'Pressure (millibars)'
        ]
        
        # 时间特征列（基于UTC时间）
        time_feature_columns = [
            'Hour', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Season',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
            'DayOfWeek_sin', 'DayOfWeek_cos'
        ]
        
        # 编码后的字符串特征列
        encoded_string_columns = [f'{col}_encoded' for col in string_columns if col in processed_df.columns]
        
        # 合并所有特征列
        all_feature_columns = []
        
        # 添加存在的数值列
        for col in numeric_columns:
            if col in processed_df.columns:
                all_feature_columns.append(col)
        
        # 添加存在的时间特征列
        for col in time_feature_columns:
            if col in processed_df.columns:
                all_feature_columns.append(col)
        
        # 添加存在的编码字符串列
        for col in encoded_string_columns:
            if col in processed_df.columns:
                all_feature_columns.append(col)
        
        print(f"选择的特征列 ({len(all_feature_columns)} 个): {all_feature_columns}")
        
        # 4. 数据清理和处理缺失值
        for col in all_feature_columns:
            if col in processed_df.columns:
                # 处理缺失值
                if processed_df[col].isna().sum() > 0:
                    if col in encoded_string_columns:
                        # 字符串编码列用0填充
                        processed_df[col] = processed_df[col].fillna(0)
                    else:
                        # 数值列用中位数填充
                        median_val = processed_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0
                        processed_df[col] = processed_df[col].fillna(median_val)
                
                    print(f"  {col}: 填充了 {processed_df[col].isna().sum()} 个缺失值")
                
                # 处理无穷值
                inf_count = np.isinf(processed_df[col]).sum()
                if inf_count > 0:
                    median_val = processed_df[col].replace([np.inf, -np.inf], np.nan).median()
                    if pd.isna(median_val):
                        median_val = 0
                    processed_df[col] = processed_df[col].replace([np.inf, -np.inf], median_val)
                    print(f"  {col}: 处理了 {inf_count} 个无穷值")
        
        # 5. 特征标准化
        # 不需要标准化的列（分类特征和已经标准化的周期特征）
        no_scale_columns = (
            encoded_string_columns + 
            ['Hour', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Season'] +
            ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']
        )
        
        for col in all_feature_columns:
            if col not in no_scale_columns and col in processed_df.columns:
                # 只对连续数值特征进行标准化
                if processed_df[col].std() > 1e-6:  # 避免标准差为0
                    scaler = StandardScaler()
                    values = processed_df[col].values.reshape(-1, 1)
                    processed_df[col] = scaler.fit_transform(values).flatten()
                    self.scalers[col] = scaler
                    print(f"  标准化: {col}")
        
        # 6. 准备最终数据
        final_data = processed_df[all_feature_columns].copy()
        
        # 最终检查和清理
        final_data = final_data.replace([np.inf, -np.inf], 0)
        final_data = final_data.fillna(0)
        
        # 数据统计
        print(f"\n数据预处理完成:")
        print(f"最终数据形状: {final_data.shape}")
        print(f"特征列数量: {len(all_feature_columns)}")
        print(f"包含NaN: {final_data.isna().sum().sum()}")
        print(f"包含Inf: {np.isinf(final_data.values).sum()}")
        
        # 显示每种类型特征的数量
        numeric_count = len([c for c in all_feature_columns if c in numeric_columns])
        time_count = len([c for c in all_feature_columns if c in time_feature_columns])
        string_count = len([c for c in all_feature_columns if c in encoded_string_columns])
        
        print(f"特征类型分布:")
        print(f"  原始数值特征: {numeric_count}")
        print(f"  时间特征 (基于UTC): {time_count}")
        print(f"  编码字符串特征: {string_count}")
        
        return final_data.values, final_data, all_feature_columns

    def _convert_to_utc(self, date_string):
        """手动将带时区的时间字符串转换为UTC时间"""
        import re
        
        try:
            # 处理格式如 "2006-04-01 00:00:00.000 +0200"
            # 使用正则表达式提取时区信息
            pattern = r'(.+?)\s*([+-]\d{4})$'
            match = re.match(pattern, str(date_string).strip())
            
            if match:
                datetime_part = match.group(1)
                timezone_part = match.group(2)
                
                # 解析基础时间
                base_time = pd.to_datetime(datetime_part, errors='coerce')
                
                if pd.isna(base_time):
                    return None
                
                # 提取时区偏移（小时数）
                sign = 1 if timezone_part[0] == '+' else -1
                hours_offset = int(timezone_part[1:3])
                minutes_offset = int(timezone_part[3:5])
                
                # 计算总的分钟偏移
                total_minutes_offset = sign * (hours_offset * 60 + minutes_offset)
                
                # 转换为UTC：减去时区偏移
                utc_time = base_time - pd.Timedelta(minutes=total_minutes_offset)
                
                return utc_time
            else:
                # 如果没有时区信息，尝试直接解析
                return pd.to_datetime(date_string, errors='coerce')
                
        except Exception as e:
            print(f"转换时间 '{date_string}' 时出错: {e}")
            return None
    
    def _create_default_time_features(self, processed_df):
        """创建默认的时间特征"""
        print("创建默认时间特征...")
        
        # 创建简单的时间特征
        n_rows = len(processed_df)
        
        processed_df['Hour'] = np.arange(n_rows) % 24
        processed_df['Month'] = ((np.arange(n_rows) // (24 * 30)) % 12) + 1
        processed_df['Day'] = ((np.arange(n_rows) // 24) % 30) + 1
        processed_df['DayOfWeek'] = ((np.arange(n_rows) // 24) % 7)
        processed_df['DayOfYear'] = ((np.arange(n_rows) // 24) % 365) + 1
        processed_df['Season'] = ((processed_df['Month'] - 1) // 3) % 4
        
        # 周期特征
        processed_df['Hour_sin'] = np.sin(2 * np.pi * processed_df['Hour'] / 24)
        processed_df['Hour_cos'] = np.cos(2 * np.pi * processed_df['Hour'] / 24)
        processed_df['Month_sin'] = np.sin(2 * np.pi * processed_df['Month'] / 12)
        processed_df['Month_cos'] = np.cos(2 * np.pi * processed_df['Month'] / 12)
        processed_df['DayOfWeek_sin'] = np.sin(2 * np.pi * processed_df['DayOfWeek'] / 7)
        processed_df['DayOfWeek_cos'] = np.cos(2 * np.pi * processed_df['DayOfWeek'] / 7)
        
        print("默认时间特征创建完成")

    def create_sequences(self, data, seq_length, pred_length, target_indices):
        """创建用于时间序列预测的序列数据"""
        print(f"创建序列数据: seq_length={seq_length}, pred_length={pred_length}")
        
        X, y = [], []
        
        # 确保有足够的数据来创建序列
        total_length = seq_length + pred_length
        if len(data) < total_length:
            print(f"警告: 数据长度 {len(data)} 小于所需的最小长度 {total_length}")
            return np.array([]), np.array([])
        
        # 创建滑动窗口序列
        for i in range(len(data) - total_length + 1):
            # 输入序列 (历史数据)
            X_seq = data[i:i + seq_length]
            
            # 目标序列 (未来数据，只包含目标变量)
            y_seq = data[i + seq_length:i + seq_length + pred_length]
            y_seq = y_seq[:, target_indices[0]]  # 只取第一个目标变量
            
            X.append(X_seq)
            y.append(y_seq)
        
        X = np.array(X)
        y = np.array(y)
        
        # 重塑y的形状以匹配模型期望 [batch_size, pred_length, target_dim]
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]  # 添加特征维度
        
        print(f"序列创建完成: X.shape={X.shape}, y.shape={y.shape}")
        
        return X, y

# 数据集类
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 配置类，存储实验设置
class Config:
    def __init__(self):
        # 数据设置
        self.data_path = 'weather.csv'
        self.batch_size = 512
        self.seq_length = 168  # 一周的小时数
        self.pred_length = 24  # 预测未来一天
        
        # 模型设置
        self.d_model = 128      # Transformer模型维度
        self.nhead = 8          # 注意力头数
        self.num_layers = 3     # Transformer层数
        self.dropout = 0.1      # Dropout比率
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        
        # 训练设置
        self.epochs = 20
        self.patience = 5      # 早停耐心值
        self.device = torch.device("cuda")
        
        # 实验设置
        self.target_variables = {
            'Temperature (C)': 0,        # 温度
            'Humidity': 1,               # 湿度
            'Wind Speed (km/h)': 2,      # 风速
            'Pressure (millibars)': 3    # 气压
        }

# 基础Transformer模型
class WeatherTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dropout=0.1, seq_length=168, pred_length=24, target_dim=1):
        super(WeatherTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        self.target_dim = target_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, 
            dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_dim * pred_length)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局信息汇总 (取平均或使用最后一个时间步)
        x = x.mean(dim=1)  # 全局平均池化
        
        # 预测未来序列
        output = self.output_layer(x)
        output = output.view(batch_size, self.pred_length, self.target_dim)
        
        return output

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 改进的Transformer模型 - 使用局部注意力机制
class LocalAttentionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 window_size=24, dropout=0.1, seq_length=168, pred_length=24, target_dim=1):
        super(LocalAttentionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        self.target_dim = target_dim
        self.window_size = window_size
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 局部注意力Transformer编码器
        self.encoder_layers = nn.ModuleList([
            LocalAttentionEncoderLayer(d_model, nhead, window_size, dropout) 
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_dim * pred_length)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 应用局部注意力层
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 全局信息汇总
        x = x.mean(dim=1)
        
        # 预测未来序列
        output = self.output_layer(x)
        output = output.view(batch_size, self.pred_length, self.target_dim)
        
        return output

# 局部注意力编码器层
class LocalAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size, dropout=0.1):
        super(LocalAttentionEncoderLayer, self).__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, src):
        # 应用局部注意力
        seq_len = src.size(1)
        outputs = []
        
        for i in range(0, seq_len, self.window_size):
            end_idx = min(i + self.window_size, seq_len)
            
            # 处理当前窗口
            window = src[:, i:end_idx, :]
            
            # 自注意力
            attn_output, _ = self.self_attn(window, window, window)
            window = window + self.dropout(attn_output)
            window = self.norm1(window)
            
            # 前馈网络
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(window))))
            window = window + self.dropout(ff_output)
            window = self.norm2(window)
            
            outputs.append(window)
        
        # 拼接所有窗口输出
        return torch.cat(outputs, dim=1)

# CNN-Transformer混合模型
class CNNTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 cnn_out_channels=64, kernel_size=3,
                 dropout=0.1, seq_length=168, pred_length=24, target_dim=1):
        super(CNNTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        self.target_dim = target_dim
        
        # CNN层用于特征提取
        self.conv1 = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        
        # CNN到Transformer的投影层
        self.projection = nn.Linear(cnn_out_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, 
            dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_dim * pred_length)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN特征提取 - 需要转置以匹配CNN输入格式 [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)  
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = self.pool(x_cnn)
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.pool(x_cnn)
        
        # 转换回 [batch, seq_len, features]
        x_cnn = x_cnn.transpose(1, 2)
        
        # 投影到Transformer维度
        x = self.projection(x_cnn)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局信息汇总
        x = x.mean(dim=1)
        
        # 预测未来序列
        output = self.output_layer(x)
        output = output.view(batch_size, self.pred_length, self.target_dim)
        
        return output

# LSTM模型 (用于对比)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 dropout=0.1, pred_length=24, target_dim=1):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, target_dim * pred_length)
        )
        
        self.pred_length = pred_length
        self.target_dim = target_dim
        
    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.output_layer(last_output)
        output = output.view(batch_size, self.pred_length, self.target_dim)
        return output
    
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 训练器类
class Trainer:
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epoch_times = []
        
    def train_epoch(self, train_loader):
        """训练一个epoch，增加NaN检测和处理"""
        start_time = time.time()
    
        self.model.train()
        total_loss = 0
        num_batches = 0
        nan_batches = 0
    
        for batch_x, batch_y in train_loader:
        # 检查输入数据是否包含NaN/Inf
            if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                print("警告: 输入批次包含NaN或Inf值，已跳过此批次")
                nan_batches += 1
                continue
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
        
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
        
        # 检查损失是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 批次 {num_batches} 中检测到NaN/Inf损失，跳过此批次")
                nan_batches += 1
                continue
        
            loss.backward()
        # 梯度裁剪以防止梯度爆炸（可以尝试降低max_norm值）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        
            total_loss += loss.item()
            num_batches += 1
    
        if num_batches == 0:
            print(f"警告: 所有批次({nan_batches}个)都出现了NaN/Inf损失或包含异常值，返回NaN损失")
            avg_loss = float('nan')
        else:
            avg_loss = total_loss / num_batches
            if nan_batches > 0:
                print(f"信息: 跳过了 {nan_batches} 个包含NaN/Inf的批次")
    
        self.train_losses.append(avg_loss)
    
        epoch_time = time.time() - start_time
        self.epoch_times.append(epoch_time)
    
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型，增加NaN检测和处理"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        nan_batches = 0
    
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # 检查输入数据是否包含NaN/Inf
                if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                    nan_batches += 1
                    continue
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
            
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            # 检查损失是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    continue
            
                total_loss += loss.item()
                num_batches += 1
    
        if num_batches == 0:
            print(f"警告: 所有验证批次({nan_batches}个)都出现了NaN/Inf损失或包含异常值，返回NaN损失")
            avg_loss = float('nan')
        else:
            avg_loss = total_loss / num_batches
            if nan_batches > 0:
                print(f"信息: 验证时跳过了 {nan_batches} 个包含NaN/Inf的批次")
    
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=20, patience=5):
        """训练模型，改进NaN处理策略"""
        best_val_loss = float('inf')
        patience_counter = 0
        nan_epochs = 0
        
        print(f"开始训练，epochs: {epochs}")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 检查是否出现NaN损失
            if np.isnan(train_loss) or np.isnan(val_loss):
                nan_epochs += 1
                print(f"警告: Epoch {epoch+1}/{epochs} 出现NaN损失。Train Loss: {train_loss}, Val Loss: {val_loss}")
                
                # 如果连续出现NaN，降低学习率
                if nan_epochs >= 2:
                    new_lr = current_lr * 0.5
                    print(f"连续出现NaN损失，降低学习率: {current_lr:.6f} -> {new_lr:.6f}")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    current_lr = new_lr
                
                # 如果学习率已经很小或多次出现NaN，考虑提前停止
                if current_lr < 1e-6 or nan_epochs > 4:
                    print(f"多次出现NaN损失或学习率过小，提前停止训练")
                    break
                    
                self.learning_rates.append(current_lr)
                patience_counter += 1
                continue
            else:
                nan_epochs = 0  # 重置NaN计数器
            
            # 正常更新学习率调度器
            self.scheduler.step(val_loss)
            self.learning_rates.append(current_lr)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {current_lr:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        if not np.isinf(best_val_loss) and not np.isnan(best_val_loss):
            try:
                self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
                print("加载最佳模型")
            except:
                print("使用当前模型")
        else:
            print("训练过程中未找到有效模型，使用当前模型")
    
        return self.model
    
    def plot_training_curves(self, model_name="Model"):
        """绘制训练曲线并保存到文件"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, 'b-', label='训练损失')
        plt.plot(self.val_losses, 'r-', label='验证损失')
        plt.xlabel('Epoch', fontproperties=myfont)
        plt.ylabel('Loss', fontproperties=myfont)
        plt.title('训练和验证损失', fontproperties=myfont)
        plt.legend(prop=myfont)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates, 'g-')
        plt.title('学习率变化', fontproperties=myfont)
        plt.xlabel('Epoch', fontproperties=myfont)
        plt.ylabel('Learning Rate', fontproperties=myfont)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像到根目录，使用模型名称
        clean_model_name = model_name.replace(" ", "_").replace("-", "_")
        filename = f"Training_{clean_model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图像，不显示
        print(f"训练曲线已保存到: {filename}")

# 评估器类
class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def predict(self, test_loader):
        """进行预测"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        return predictions, actuals
    
    def calculate_metrics(self, predictions, actuals):
        """计算评估指标"""
        pred_flat = predictions.reshape(-1)
        actual_flat = actuals.reshape(-1)
        
        mse = mean_squared_error(actual_flat, pred_flat)
        mae = mean_absolute_error(actual_flat, pred_flat)
        rmse = np.sqrt(mse)
        
        try:
            r2 = r2_score(actual_flat, pred_flat)
        except:
            r2 = 0.0
        
        # 计算MAPE（平均绝对百分比误差）
        mask = actual_flat != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual_flat[mask] - pred_flat[mask]) / actual_flat[mask])) * 100
        else:
            mape = 0.0
        
        # 计算方向准确度（预测趋势是否正确）
        if len(actual_flat) > 1:
            actual_diff = np.diff(actual_flat)
            pred_diff = np.diff(pred_flat)
            direction_accuracy = np.mean((actual_diff * pred_diff) > 0) * 100
        else:
            direction_accuracy = 0.0
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }
        
        return metrics
    
    def plot_predictions(self, predictions, actuals, title="预测结果对比", model_name="Model", target_var="Variable"):
        """绘制预测结果并保存到文件"""
        n_samples = min(5, predictions.shape[0])
        
        plt.figure(figsize=(15, 10))
        
        # 绘制样本预测
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i+1)
            plt.plot(predictions[i, :, 0], 'r-', label='预测值')
            plt.plot(actuals[i, :, 0], 'b-', label='真实值')
            plt.title(f'样本 {i+1} 预测对比', fontproperties=myfont)
            plt.legend(prop=myfont)
            plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        
        # 保存预测对比图，使用模型名称和目标变量
        clean_model_name = model_name.replace(" ", "_").replace("-", "_")
        clean_target_var = target_var.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        filename1 = f"Predictions_{clean_model_name}_{clean_target_var}.png"
        plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图像，不显示
        print(f"预测对比图已保存到: {filename1}")
        
        # 绘制性能指标
        metrics = self.calculate_metrics(predictions, actuals)
        
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12, fontproperties=myfont)
        plt.axis('off')
        plt.title('模型性能指标', fontproperties=myfont)
        plt.tight_layout()
        
        # 保存性能指标图，使用模型名称和目标变量
        filename2 = f"Metrics_{clean_model_name}_{clean_target_var}.png"
        plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图像，不显示
        print(f"性能指标图已保存到: {filename2}")
        
        return metrics

# 超参数调优类
class HyperparameterTuner:
    def __init__(self, config, feature_dim, target_var, train_data, val_data):
        self.config = config
        self.feature_dim = feature_dim
        self.target_var = target_var
        self.train_data = train_data
        self.val_data = val_data
        self.results = []
        
    def tune_model(self, model_class, param_grid, epochs=20):
        """对模型进行超参数调优"""
        print(f"开始对 {model_class.__name__} 进行超参数调优")
        
        for params in self._generate_param_combinations(param_grid):
            print(f"\n尝试参数: {params}")
            
            # 分离模型参数和训练参数
            model_params = {}
            training_params = {}
            
            for key, value in params.items():
                if key in ['learning_rate', 'weight_decay']:
                    # 这些是训练器参数，不是模型参数
                    training_params[key] = value
                else:
                    # 这些是模型参数
                    model_params[key] = value
        
            # 创建模型
            if model_class == LSTMModel:
                # LSTM模型参数处理
                final_model_params = {
                    'input_dim': self.feature_dim,
                    'pred_length': self.config.pred_length,
                    'target_dim': 1
                }
                # 添加LSTM特有参数
                lstm_params = ['hidden_dim', 'num_layers', 'dropout']
                for param in lstm_params:
                    if param in model_params:
                        final_model_params[param] = model_params[param]
            else:
                # Transformer类模型参数处理
                final_model_params = {
                    'input_dim': self.feature_dim,
                    'pred_length': self.config.pred_length,
                    'target_dim': 1
                }
                # 添加模型特定参数
                final_model_params.update(model_params)
        
            model = model_class(**final_model_params)
        
            # 训练模型
            trainer = Trainer(
                model, 
                self.config.device, 
                learning_rate=training_params.get('learning_rate', self.config.learning_rate),
                weight_decay=training_params.get('weight_decay', self.config.weight_decay)
            )
        
            model = trainer.train(
                self.train_data, 
                self.val_data, 
                epochs=epochs, 
                patience=5
            )
        
            # 评估模型
            evaluator = Evaluator(model, self.config.device)
            predictions, actuals = evaluator.predict(self.val_data)
            metrics = evaluator.calculate_metrics(predictions, actuals)
        
            # 记录结果
            result = {
                'params': params,
                'metrics': metrics,
                'val_loss': trainer.val_losses[-1] if trainer.val_losses else float('inf')
            }
        
            self.results.append(result)
            print(f"结果: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
    
        # 根据验证损失排序结果
        self.results.sort(key=lambda x: x['val_loss'])
    
        return self.results
    
    def _generate_param_combinations(self, param_grid):
        """生成参数组合"""
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def plot_tuning_results(self, model_name="Model"):
        """绘制调优结果并保存到文件"""
        if not self.results:
            print("没有调优结果可绘制")
            return
        
        # 找出关键参数和性能指标
        params = list(self.results[0]['params'].keys())
        metrics = ['RMSE', 'MAE', 'R2']
        
        n_params = len(params)
        
        plt.figure(figsize=(15, n_params * 4))
        
        for i, param in enumerate(params):
            for j, metric in enumerate(metrics):
                plt.subplot(n_params, len(metrics), i * len(metrics) + j + 1)
                
                x_values = [result['params'][param] for result in self.results]
                y_values = [result['metrics'][metric] for result in self.results]
                
                plt.scatter(x_values, y_values, alpha=0.7)
                plt.xlabel(param, fontproperties=myfont)
                plt.ylabel(metric, fontproperties=myfont)
                plt.title(f'{param} vs {metric}', fontproperties=myfont)
                plt.grid(True, alpha=0.3)
                
                # 对于学习率等对数尺度参数
                if param in ['learning_rate', 'd_model']:
                    plt.xscale('log')
        
        plt.tight_layout()
        
        # 保存调优结果图，使用模型名称
        clean_model_name = model_name.replace(" ", "_").replace("-", "_")
        filename = f"Hyperparameter_Tuning_{clean_model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图像，不显示
        print(f"超参数调优结果已保存到: {filename}")

import itertools
import os
from sklearn.model_selection import train_test_split

# 实验类
class WeatherExperiment:
    def __init__(self, config):
        self.config = config
        self.data_processor = WeatherDataProcessor()
        self.data = None
        self.df = None
        self.feature_cols = None
        self.target_variables = config.target_variables
        self.model_results = {}
        
        # 序列设置
        self.seq_length = config.seq_length
        self.pred_length = config.pred_length
        
        # 模型设置
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        
    def load_data(self):
        """加载和预处理数据"""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件 {self.config.data_path} 不存在")
            
        self.data, self.df, self.feature_cols = self.data_processor.load_and_preprocess_data(
            self.config.data_path
        )
        
        print(f"数据加载完成，形状: {self.data.shape}")
        print(f"特征列: {self.feature_cols}")
        return self.data
    
    def prepare_target_data(self, target_var_name, test_size=0.2):
        """准备特定目标变量的训练数据，增加数据验证"""
        if target_var_name not in self.target_variables:
            raise ValueError(f"目标变量 {target_var_name} 不在配置的目标变量中")
            
        if self.data is None:
            self.load_data()
            
        # 获取目标变量的索引
        target_idx = [self.feature_cols.index(target_var_name)]
        
        # 创建序列
        X, y = self.data_processor.create_sequences(
            self.data, 
            self.seq_length, 
            self.pred_length, 
            target_idx
        )
        
        # 检查并处理异常值
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"警告: 输入序列X中存在NaN或Inf值 ({np.isnan(X).sum() + np.isinf(X).sum()} 个)，已替换为0")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
        if np.isnan(y).any() or np.isinf(y).any():
            print(f"警告: 目标序列y中存在NaN或Inf值 ({np.isnan(y).sum() + np.isinf(y).sum()} 个)，已替换为0")
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
    
        # 检查序列长度
        if len(X) == 0 or len(y) == 0:
            raise ValueError(f"无法为 {target_var_name} 创建有效的序列。请检查数据长度和序列设置。")
    
        print(f"创建了 {len(X)} 个序列，每个序列长度为 {self.seq_length}，预测长度为 {self.pred_length}")
    
        # 分割数据
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, shuffle=False
        )
        
        # 创建数据集
        train_dataset = WeatherDataset(X_train, y_train)
        val_dataset = WeatherDataset(X_val, y_val)
        test_dataset = WeatherDataset(X_test, y_test)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        print(f"数据准备完成:")
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def run_model_experiment(self, model_class, model_name, target_var_name, **model_kwargs):
        """运行单个模型的实验"""
        print(f"\n开始 {model_name} 在 {target_var_name} 上的实验")
        
        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_target_data(target_var_name)
        
        # 创建模型
        input_dim = self.data.shape[1]
        
        # 为不同模型类型设置不同的参数
        if model_class == LSTMModel:
            # LSTM模型不使用transformer相关参数
            model_params = {
                'input_dim': input_dim,
                'pred_length': self.pred_length,
                'target_dim': 1
            }
            # 只添加LSTM特有的参数
            lstm_specific_params = ['hidden_dim', 'num_layers', 'dropout']
            for param in lstm_specific_params:
                if param in model_kwargs:
                    model_params[param] = model_kwargs[param]
        
            model = model_class(**model_params)
        else:
            # Transformer类模型使用标准参数
            model_params = {
                'input_dim': input_dim,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'pred_length': self.pred_length,
                'target_dim': 1
            }
            # 添加额外的模型特定参数
            model_params.update(model_kwargs)
            
            model = model_class(**model_params)
    
        # 训练模型
        trainer = Trainer(model, self.config.device, self.config.learning_rate, self.config.weight_decay)
        model = trainer.train(train_loader, val_loader, self.config.epochs, self.config.patience)
        
        # 评估模型
        evaluator = Evaluator(model, self.config.device)
        test_predictions, test_actuals = evaluator.predict(test_loader)
        metrics = evaluator.calculate_metrics(test_predictions, test_actuals)
        
        # 保存结果
        result = {
            'model': model,
            'trainer': trainer,
            'metrics': metrics,
            'predictions': test_predictions,
            'actuals': test_actuals
        }
        
        # 存储结果
        if target_var_name not in self.model_results:
            self.model_results[target_var_name] = {}
        
        self.model_results[target_var_name][model_name] = result
        
        # 显示训练曲线和预测结果
        trainer.plot_training_curves(model_name)
        evaluator.plot_predictions(test_predictions, test_actuals, 
                                  title=f"{model_name} 在 {target_var_name} 上的预测",
                                  model_name=model_name, 
                                  target_var=target_var_name)
        
        print(f"\n{model_name} 在 {target_var_name} 上的性能:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
            
        return result
    
    def compare_models(self, target_var_name):
        """比较不同模型在特定目标变量上的性能并保存图像"""
        if target_var_name not in self.model_results:
            print(f"没有 {target_var_name} 的模型结果")
            return
            
        models = self.model_results[target_var_name]
        
        if not models:
            print("没有模型可比较")
            return
            
        # 比较关键指标
        metrics_to_compare = ['RMSE', 'MAE', 'R2', 'MAPE', 'Direction_Accuracy']
        
        # 准备数据
        model_names = list(models.keys())
        metrics_data = {metric: [] for metric in metrics_to_compare}
        
        for model_name, result in models.items():
            for metric in metrics_to_compare:
                metrics_data[metric].append(result['metrics'][metric])
    
        # 绘制比较图
        plt.figure(figsize=(15, 12))
        
        for i, metric in enumerate(metrics_to_compare):
            plt.subplot(3, 2, i+1)
            bars = plt.bar(model_names, metrics_data[metric])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0, fontproperties=myfont)
            
            plt.title(f'{metric} 比较', fontproperties=myfont)
            plt.ylabel(metric, fontproperties=myfont)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'{target_var_name} 上的模型性能比较', fontsize=16, y=1.02, fontproperties=myfont)
        
        # 保存模型比较图 - 使用更清晰的文件名
        clean_target_var = target_var_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        filename = f"Model_Comparison_{clean_target_var}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图像，不显示
        print(f"模型比较图已保存到: {filename}")
        
        # 表格形式显示所有指标
        print(f"\n{target_var_name} 上的模型性能比较:")
        header = "模型名称".ljust(20) + " | " + " | ".join([metric.ljust(10) for metric in metrics_to_compare])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for model_name in model_names:
            metrics = models[model_name]['metrics']
            row = model_name.ljust(20) + " | " + " | ".join([f"{metrics[metric]:.4f}".ljust(10) for metric in metrics_to_compare])
            print(row)
        
        print("-" * len(header))
    
    def run_hyperparameter_tuning(self, model_class, target_var_name, param_grid):
        """运行超参数调优"""
        # 准备数据
        train_loader, val_loader, _ = self.prepare_target_data(target_var_name)
        
        # 创建调优器
        tuner = HyperparameterTuner(
            self.config,
            self.data.shape[1],
            target_var_name,
            train_loader,
            val_loader
        )
        
        # 运行调优
        results = tuner.tune_model(model_class, param_grid, epochs=20)
        
        # 显示最佳参数
        print(f"\n超参数调优结果:")
        print(f"最佳参数: {results[0]['params']}")
        print(f"最佳验证损失: {results[0]['val_loss']:.6f}")
        print(f"最佳指标: RMSE={results[0]['metrics']['RMSE']:.4f}, R2={results[0]['metrics']['R2']:.4f}")
        
        # 绘制调优结果 - 传递模型名称
        tuner.plot_tuning_results(model_class.__name__)
        
        return results[0]['params']

# 主函数
def main():
    # 创建配置
    config = Config()
    
    # 创建实验
    experiment = WeatherExperiment(config)
    
    # 加载数据
    experiment.load_data()
    
    # 选择要预测的目标变量
    target_var = 'Temperature (C)'
    
    # 运行基础Transformer模型
    experiment.run_model_experiment(
        WeatherTransformer, 
        "基础Transformer", 
        target_var
    )
    
    # 运行带局部注意力的Transformer模型
    experiment.run_model_experiment(
        LocalAttentionTransformer, 
        "局部注意力Transformer", 
        target_var,
        window_size=24
    )
    
    # 运行CNN-Transformer混合模型
    experiment.run_model_experiment(
        CNNTransformer, 
        "CNN-Transformer", 
        target_var
    )
    
    # 运行LSTM模型(用于对比) - 修复参数传递
    experiment.run_model_experiment(
        LSTMModel, 
        "LSTM", 
        target_var,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    
    # 比较不同模型的性能
    experiment.compare_models(target_var)
    
    # 对其他气象变量进行测试
    for var_name in ['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']:
        if var_name in experiment.feature_cols:
            # 使用最佳模型对其他变量进行测试
            experiment.run_model_experiment(
                CNNTransformer,  # 根据之前的对比选择最佳模型
                "CNN-Transformer", 
                var_name
            )
    
    # 超参数调优示例 - 修复参数网格
    param_grid = {
        'd_model': [128,256],
        'nhead': [4, 8],
        'num_layers': [2, 3],
        'learning_rate': [0.0001, 0.0002],  # 这将被正确处理为训练参数
        'dropout': [0.1, 0.2]
    }
    
    best_params = experiment.run_hyperparameter_tuning(
        WeatherTransformer,
        target_var,
        param_grid
    )
    
    # 使用最佳参数运行模型 - 分离模型参数和训练参数
    model_params = {k: v for k, v in best_params.items() if k not in ['learning_rate', 'weight_decay']}
    
    experiment.run_model_experiment(
        WeatherTransformer, 
        "调优后的Transformer", 
        target_var,
        **model_params
    )
    
    # 最终比较所有模型
    experiment.compare_models(target_var)

if __name__ == "__main__":
    main()