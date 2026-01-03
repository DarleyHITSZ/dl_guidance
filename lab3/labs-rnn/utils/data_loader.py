import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

proxy = 'http://127.0.0.1:7890'    
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
class DataLoader:
    def __init__(self, stock_symbol='AAPL', start_date='2010-01-01', end_date='2023-12-31', seq_len=30, batch_size=32, test_split=0.2):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.test_split = test_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_dir = 'data'
        
    def download_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        data_file = os.path.join(self.data_dir, f'{self.stock_symbol}.csv')
        
        if os.path.exists(data_file):
            print(f"Loading existing data from {data_file}")
            try:
                # 尝试读取完整数据，检查Date列是否存在
                df = pd.read_csv(data_file)
                
                if 'Date' not in df.columns:
                    # 处理特殊格式：跳过第2行（Ticker信息）和第3行（空Date行）
                    print("Detected special CSV format, skipping invalid rows...")
                    df = pd.read_csv(data_file, skiprows=[1, 2])
                    
                    # 重新设置列名
                    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                    
                    # 确保Date列是日期类型
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    data = df
                else:
                    # 正常格式，直接读取
                    data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
            except Exception as e:
                print(f"Error reading CSV file: {e}. Re-downloading...")
                # 删除有问题的文件并重新下载
                os.remove(data_file)
                return self.download_data()
        else:
            print(f"Downloading {self.stock_symbol} stock data...")
            data = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
            data.to_csv(data_file)
            print(f"Data saved to {data_file}")
        
        return data
    
    def preprocess_data(self, data, feature_cols=['Close']):
        self.feature_cols = feature_cols
        self.input_size = len(feature_cols)
        
        # 选择特征列
        features = data[feature_cols].values
        
        # 归一化
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features
    
    def create_sequences(self, scaled_features):
        X = []
        y = []
        
        for i in range(len(scaled_features) - self.seq_len):
            X.append(scaled_features[i:i+self.seq_len])
            y.append(scaled_features[i+self.seq_len, 0])  # 预测下一个时间步的第一个特征（Close）
        
        X = np.array(X)
        y = np.array(y)
        
        # 调整维度为 (seq_len, input_size, batch_size) 用于模型输入
        X = np.transpose(X, (1, 2, 0))  # (samples, seq_len, input_size) -> (seq_len, input_size, samples)
        y = y.reshape(1, -1)  # (samples,) -> (1, samples)
        
        return X, y
    
    def split_data(self, X, y):
        total_samples = X.shape[2]
        test_samples = int(total_samples * self.test_split)
        
        X_train = X[:, :, :-test_samples]
        y_train = y[:, :-test_samples]
        X_test = X[:, :, -test_samples:]
        y_test = y[:, -test_samples:]
        
        return X_train, y_train, X_test, y_test
    
    def create_batches(self, X, y):
        total_samples = X.shape[2]
        num_batches = total_samples // self.batch_size
        
        X_batches = []
        y_batches = []
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            
            X_batch = X[:, :, start_idx:end_idx]
            y_batch = y[:, start_idx:end_idx]
            
            X_batches.append(X_batch)
            y_batches.append(y_batch)
        
        return X_batches, y_batches
    
    def load_and_preprocess(self, feature_cols=['Close']):
        # 下载数据
        raw_data = self.download_data()
        
        # 预处理数据
        scaled_features = self.preprocess_data(raw_data, feature_cols)
        
        # 创建序列数据
        X, y = self.create_sequences(scaled_features)
        
        # 划分训练测试集
        X_train, y_train, X_test, y_test = self.split_data(X, y)
        
        # 创建批次
        X_train_batches, y_train_batches = self.create_batches(X_train, y_train)
        X_test_batches, y_test_batches = self.create_batches(X_test, y_test)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_batches': X_train_batches,
            'y_train_batches': y_train_batches,
            'X_test_batches': X_test_batches,
            'y_test_batches': y_test_batches,
            'scaler': self.scaler,
            'input_size': self.input_size
        }
    
    def inverse_transform(self, data):
        # 将预测值转换回原始尺度
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            if data.shape[0] == 1:
                data = data.T
        
        # 为所有特征列填充与第一列相同的值，然后取第一列
        full_data = np.zeros((data.shape[0], self.input_size))
        full_data[:, 0] = data[:, 0]
        
        inverse_transformed = self.scaler.inverse_transform(full_data)
        return inverse_transformed[:, 0]