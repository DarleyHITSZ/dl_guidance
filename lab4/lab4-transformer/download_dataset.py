import os
import requests
import zipfile
import shutil
import random
from tqdm import tqdm
from typing import List, Tuple

class WMT14DatasetDownloader:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.urls = {
            'train': 'https://storage.googleapis.com/tensorflow-nmt/data/wmt16_en_de.tar.gz',
            'test': 'https://storage.googleapis.com/tensorflow-nmt/data/newstest2014.en',
            'test_de': 'https://storage.googleapis.com/tensorflow-nmt/data/newstest2014.de'
        }
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_file(self, url: str, filename: str) -> None:
        """下载文件，支持断点续传"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            print(f"文件 {filename} 已存在，跳过下载")
            return
            
        headers = {}
        file_size = 0
        # 检查是否已存在部分文件
        if os.path.exists(f"{filepath}.part"):
            file_size = os.path.getsize(f"{filepath}.part")
            headers['Range'] = f'bytes={file_size}-'
            print(f"继续下载 {filename}，已下载 {file_size} 字节")
        
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0)) + file_size
        mode = 'ab' if file_size > 0 else 'wb'
        
        with open(f"{filepath}.part", mode) as file, tqdm(total=total_size, unit='iB', unit_scale=True, initial=file_size) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                pbar.update(size)
        
        # 下载完成后重命名
        os.rename(f"{filepath}.part", filepath)
        print(f"文件 {filename} 下载完成")
        
    def extract_file(self, filename: str) -> None:
        """解压文件"""
        filepath = os.path.join(self.data_dir, filename)
        extract_dir = os.path.join(self.data_dir, filename.split('.')[0])
        
        if os.path.exists(extract_dir):
            print(f"文件 {filename} 已解压，跳过解压")
            return
            
        print(f"解压文件 {filename}")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
    def load_sample_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """加载样本数据集（100条训练+10条测试）"""
        # 德英句子对样本
        sample_pairs = [
            ("Dies ist ein Beispielsatz auf Deutsch.", "This is an example sentence in German."),
            ("Ich liebe die Programmierung mit Python.", "I love programming with Python."),
            ("Der Himmel ist blau und die Sonne scheint.", "The sky is blue and the sun is shining."),
            ("Python ist eine beliebte Programmiersprache.", "Python is a popular programming language."),
            ("Heute ist ein schöner Tag zum Lernen.", "Today is a beautiful day for learning."),
            ("Ich möchte einen Kaffee trinken.", "I would like to drink a coffee."),
            ("Die Katze sitzt auf dem Sofa.", "The cat is sitting on the sofa."),
            ("Was machst du in deiner Freizeit?", "What do you do in your free time?"),
            ("Ich lese gerne Bücher und sehe Filme.", "I like reading books and watching movies."),
            ("Gestern war ich im Kino.", "Yesterday I was at the cinema."),
            ("Morgen fahre ich nach Berlin.", "Tomorrow I'm driving to Berlin."),
            ("Die Schule beginnt um 8 Uhr.", "School starts at 8 o'clock."),
            ("Ich habe drei Schwestern und einen Bruder.", "I have three sisters and one brother."),
            ("Das Wetter ist heute sehr heiß.", "The weather is very hot today."),
            ("Ich muss meine Hausaufgaben machen.", "I have to do my homework."),
            ("Kannst du mir bitte helfen?", "Can you please help me?"),
            ("Ich verstehe nicht, was du sagst.", "I don't understand what you're saying."),
            ("Bitte sprechen Sie langsamer.", "Please speak more slowly."),
            ("Vielen Dank für deine Hilfe.", "Thank you very much for your help."),
            ("Es tut mir leid, ich bin spät dran.", "I'm sorry, I'm late."),
        ]
        
        # 扩展样本到100条训练数据
        train_data = []
        for i in range(5):  # 复制5次以生成100条数据
            for de, en in sample_pairs:
                # 添加一些变体
                if i % 2 == 0:
                    train_data.append((de, en))
                else:
                    # 添加轻微变化
                    de_var = de.replace('.', '!') if i % 3 == 0 else de
                    en_var = en.replace('.', '!') if i % 3 == 0 else en
                    train_data.append((de_var, en_var))
        
        # 随机选择10条作为测试数据
        random.shuffle(train_data)
        test_data = train_data[:10]
        train_data = train_data[10:110]  # 取100条训练数据
        
        return train_data, test_data
        
    def preprocess_data(self, data: List[Tuple[str, str]], min_len: int = 5, max_len: int = 50) -> List[Tuple[str, str]]:
        """数据预处理：过滤长度不合适的句子"""
        processed = []
        for de, en in data:
            de_len = len(de.split())
            en_len = len(en.split())
            if min_len <= de_len <= max_len and min_len <= en_len <= max_len:
                processed.append((de.strip(), en.strip()))
        return processed
        
    def save_processed_data(self, train_data: List[Tuple[str, str]], test_data: List[Tuple[str, str]]) -> None:
        """保存预处理后的数据集"""
        train_file = os.path.join(self.data_dir, 'train.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for de, en in train_data:
                f.write(f"{de} ||| {en}\n")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            for de, en in test_data:
                f.write(f"{de} ||| {en}\n")
        
        print(f"训练数据已保存到 {train_file}，共 {len(train_data)} 条")
        print(f"测试数据已保存到 {test_file}，共 {len(test_data)} 条")
        
    def load_processed_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """加载预处理后的数据集"""
        train_file = os.path.join(self.data_dir, 'train.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|||' in line:
                    de, en = line.strip().split('|||')
                    train_data.append((de.strip(), en.strip()))
        
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|||' in line:
                    de, en = line.strip().split('|||')
                    test_data.append((de.strip(), en.strip()))
        
        return train_data, test_data
        
    def run(self) -> None:
        """执行数据集下载和预处理流程"""
        print("开始WMT14德英数据集处理...")
        
        # 检查是否已存在处理好的数据
        if os.path.exists(os.path.join(self.data_dir, 'train.txt')) and os.path.exists(os.path.join(self.data_dir, 'test.txt')):
            print("已存在处理好的数据集，跳过处理")
            return
        
        # 注意：由于完整WMT14数据集较大，这里使用样本数据进行演示
        print("使用样本数据进行演示（100条训练+10条测试）")
        train_data, test_data = self.load_sample_data()
        
        # 预处理数据
        train_data = self.preprocess_data(train_data)
        test_data = self.preprocess_data(test_data)
        
        # 保存处理后的数据
        self.save_processed_data(train_data, test_data)
        
        print("数据集处理完成！")

if __name__ == "__main__":
    downloader = WMT14DatasetDownloader()
    downloader.run()
