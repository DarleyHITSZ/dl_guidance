import os
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
proxy = 'http://127.0.0.1:7890'    
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
class DataLoader:
    def __init__(self, data_dir='data'):
        """初始化 DataLoader 实例
        
        参数：
            data_dir (str): 数据存储目录，默认 'data'
        """
        self.data_dir = data_dir  # 定义数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)  # 创建目录（如果不存在）

    def load_yahoo_stock(self, ticker='AAPL', start_date='2011-01-01',
    end_date='2023-12-31'):
        # 1. 下载数据
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        # 2. 只保留收盘价
        data = df['Close'].values.reshape(-1, 1)
        # 3. 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        # 4. 保存数据
        data_path = os.path.join(self.data_dir,f'{ticker}_stock_data.csv')
        print(f"实际保存路径: {os.path.abspath(data_path)}")
        df.to_csv(data_path)
        return {
            'data': scaled_data,
            'scaler': scaler,
            'original_data': data
        }
        
    def create_stock_batches(self, data, seq_len, batch_size):
        """ 创建股票数据批次
        参数：
        data: 归一化后的股票数据 (n_samples, 1)
        seq_len: 序列长度（使用前 seq_len 个值预测下一个值）
        batch_size: 批次大小
        返回：
        batches: 批次列表，每个批次包含(x, y)
        x: (seq_len, input_size, batch_size)
        y: (1, batch_size)
        """
        total_len = len(data)
        x = []
        y = []
        
        # 创建序列数据
        for i in range(total_len - seq_len):
            x.append(data[i:i+seq_len]) # 前 seq_len 个数据
            y.append(data[i+seq_len]) # 下一个数据
        x = np.array(x)
        y = np.array(y)
        # 创建批次
        n_batches = len(x) // batch_size
        x = x[:n_batches * batch_size]
        y = y[:n_batches * batch_size]
        # 重塑为 (batch_size, n_batches, seq_len, 1)
        x = x.reshape(batch_size, n_batches, seq_len, 1).transpose(1, 2, 3, 0)
        y = y.reshape(batch_size, n_batches, 1).transpose(1, 2, 0)
        batches = []
        for i in range(n_batches):
            batches.append((x[i], y[i]))
        
        return batches