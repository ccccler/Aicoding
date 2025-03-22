import pandas as pd
import sqlite3
import asyncio
import aiohttp
import json
import os
from typing import List, Dict, Any
import time
from tqdm import tqdm
import argparse

class ThemeExtractor:
    def __init__(self, 
                 excel_path: str = "小量文本测试集.xlsx", 
                 api_key: str="sk-TotNw1nIUNJ6QKOvHaihightLOr68RDy1w4sXBvAasGKbUTU", 
                 db_path: str = "themes_results.db",
                 batch_size: int = 100,
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 model: str = "gpt-4o-mini",
                 system_prompt_template: str = None):
        """
        初始化主题提取器
        
        参数:
            excel_path: Excel文件路径
            api_key: OpenAI API密钥
            db_path: SQLite数据库路径
            batch_size: 异步批处理大小
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
            model: OpenAI模型名称
        """
        self.excel_path = excel_path
        self.api_key = api_key
        self.db_path = db_path
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = model
        
        # 初始化数据库
        self._init_db()
        
    def _init_db(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY,
            original_text TEXT,
            theme TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
        
    def load_data(self, text_column: str) -> pd.DataFrame:
        """
        从Excel加载数据
        
        参数:
            text_column: 包含文本数据的列名
            
        返回:
            包含ID和文本的DataFrame
        """
        df = pd.read_excel(self.excel_path)
        if text_column not in df.columns:
            raise ValueError(f"列 '{text_column}' 在Excel文件中不存在")
        
        # 确保有一个唯一ID列
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
            
        # 只保留需要的列
        return df[['id', text_column]].rename(columns={text_column: 'title'})
    
    def get_processed_ids(self) -> List[int]:
        """获取已处理的ID列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM themes")
        processed_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return processed_ids
    
    async def extract_theme(self, session: aiohttp.ClientSession, text: str, prompt_template: str,system_prompt_template: str) -> str:
        """
        使用OpenAI API提取主题
        
        参数:
            session: aiohttp会话
            text: 要分析的文本
            prompt_template: 提示词模板
            
        返回:
            提取的主题
        """
        prompt = prompt_template.format(text=text)
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    "https://xiaoai.plus/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt_template},
                            {"role": "user", "content": "经过我长时间的收集积累，#汉服资料馆#的资料数量已经上来了，如何高效率的利用这几十万份参考资料呢，分享给诸位同袍一个绝招，现在的AI基本都已经有联网搜索功能了，把你想查汉服相关内容，后面加一句，site:120.25.237.190。这是命令AI去搜汉服资料馆。从里面提取相关的信息。这样网站里的内容就任你利用了，出结果快速，且参考来源靠谱。还可以避免直接问时，AI找被污染的信源或者出现幻觉生造内容。 汉服超话#汉服##AI创造营# 收起d"},
                            {"role": "assistant", "content": '''{"topic":"AI幻觉，资料搜索，AI联网","big_topic":"AI幻觉解决方法","description":"分享解决AI搜索时出现幻觉问题的方法。"}'''},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        theme = data['choices'][0]['message']['content'].strip()
                        return theme
                    else:
                        error_text = await response.text()
                        print(f"API错误 (尝试 {attempt+1}/{self.max_retries}): {response.status} - {error_text}")
                        
                        if response.status == 429:  # 速率限制
                            wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                            print(f"达到速率限制，等待 {wait_time} 秒...")
                            await asyncio.sleep(wait_time)
                        else:
                            await asyncio.sleep(self.retry_delay)
            except Exception as e:
                print(f"请求异常 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(self.retry_delay)
                
        return "提取失败"  # 所有尝试都失败
    
    def save_result(self, item_id: int, text: str, theme: str):
        """
        将结果保存到数据库
        
        参数:
            item_id: 数据项ID
            text: 原始文本
            theme: 提取的主题
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO themes (id, original_text, theme) VALUES (?, ?, ?)",
            (item_id, text, theme)
        )
        conn.commit()
        conn.close()
    
    async def process_batch(self, 
                           batch_data: List[Dict[str, Any]], 
                           prompt_template: str,
                           system_prompt_template: str,
                           pbar: tqdm):
        """
        处理一批数据
        
        参数:
            batch_data: 要处理的数据批次
            prompt_template: 提示词模板
            system_prompt_template: 系统提示词模板
            pbar: 进度条
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in batch_data:
                task = asyncio.create_task(
                    self.extract_theme(session, item['title'], prompt_template, system_prompt_template)
                )
                tasks.append((item, task))
                
            for item, task in tasks:
                theme = await task
                self.save_result(item['id'], item['title'], theme)
                pbar.update(1)
                
    async def process_data(self, 
                          data: pd.DataFrame, 
                          prompt_template: str,
                          system_prompt_template: str,
                          limit: int = None):
        """
        处理所有数据
        
        参数:
            data: 要处理的DataFrame
            prompt_template: 提示词模板
            system_prompt_template: 系统提示词模板
            limit: 处理的最大数量
        """
        # 获取已处理的ID
        processed_ids = self.get_processed_ids()
        print(f"已找到 {len(processed_ids)} 条已处理的记录")
        
        # 过滤掉已处理的数据
        data = data[~data['id'].isin(processed_ids)]
        print(f"剩余 {len(data)} 条记录需要处理")
        
        if limit:
            data = data.head(limit)
            print(f"根据限制，将处理 {len(data)} 条记录")
            
        if len(data) == 0:
            print("没有需要处理的数据")
            return
            
        # 转换为字典列表以便处理
        data_list = data.to_dict('records')
        
        # 创建进度条
        with tqdm(total=len(data_list), desc="提取主题") as pbar:
            # 按批次处理数据
            for i in range(0, len(data_list), self.batch_size):
                batch = data_list[i:i+self.batch_size]
                await self.process_batch(batch, prompt_template, system_prompt_template, pbar)
    
    def run(self, 
            text_column: str, 
            prompt_template: str,
            system_prompt_template: str,
            limit: int = None):
        """
        运行主题提取
        
        参数:
            text_column: 文本列名
            prompt_template: 提示词模板
            system_prompt_template: 系统提示词模板
            limit: 处理的最大数量
        """
        start_time = time.time()
        print(f"开始从 '{self.excel_path}' 提取主题...")
        
        # 加载数据
        data = self.load_data(text_column)
        print(f"已加载 {len(data)} 条记录")
        
        # 运行异步处理
        asyncio.run(self.process_data(data, prompt_template, system_prompt_template, limit))
        
        elapsed = time.time() - start_time
        print(f"处理完成! 用时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    # 预设参数，无需通过命令行输入
    preset_args = {
        "excel": "nrfx/03-analy/01-input-rawdata.xlsx",  # Excel文件路径
        "column": "original_text",  # 包含文本数据的列名
        "api_key": "sk-kvH38CjwA1PTazUrgmF0WR8I1s8sjtHd6TveYTYAxddEfMQh",  # OpenAI API密钥
        "db": "nrfx/03-analy/01-themes_results.db",  # SQLite数据库路径
        "batch_size": 100,  # 异步批处理大小
        "limit": None,  # 处理的最大记录数，None表示不限制
        "model": "gpt-4o-mini"  # OpenAI模型名称
    }
    
    # 默认提示词模板
    system_prompt_template = """
    你是一个专业的内容分析编码助手，能够自动对输入内容进行分析，并提取出主题。
    我将给你一系列的社交媒体帖子，这些帖子都是有关"AI幻觉"的讨论，请你帮我分析每条帖子所讨论的内容，并且以"小主题词+大主题词+描述语句"的格式来输出。
    小主题词用来识别该语句里的所具体讨论的元素，可以是多个；
    大主题词用来总结该语句讨论的核心关键词，必须是一个；
    描述语句用来描述该语句的讨论内容，必须是一个。
    参考格式是：{"topic":"小主题词1，小主题词2，小主题词3，...","big_topic":"大主题词","description":"描述语句"}
    请你严格保持输出此json格式的内容，不要输出任何其他内容。
    """

    prompt_template = '''{text}'''
    
    # 创建并运行提取器
    extractor = ThemeExtractor(
        excel_path=preset_args["excel"],
        api_key=preset_args["api_key"],
        db_path=preset_args["db"],
        batch_size=preset_args["batch_size"],
        model=preset_args["model"],
        system_prompt_template=system_prompt_template
    )
    
    extractor.run(
        text_column=preset_args["column"],
        prompt_template=prompt_template,
        system_prompt_template=system_prompt_template,
        limit=preset_args["limit"]
    ) 