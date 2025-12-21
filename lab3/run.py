from utils.data_loader import DataLoader
import os
loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), 'data'))
#loader = DataLoader()
data = loader.load_yahoo_stock(ticker='AAPL')
print(f'Data shape: {data["data"].shape}')
print(f'First 5 data points: {data["data"][:5]}')