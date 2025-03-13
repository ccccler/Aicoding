import jieba
import pandas as pd
from gensim import corpora, models
import numpy as np
from typing import List, Dict, Any
import logging

class ChineseTopicAnalyzer:
    def __init__(self, num_topics: int = 10, passes: int = 15, minimum_probability: float = 0.01):
        """
        初始化中文主题分析器
        
        Args:
            num_topics: 主题数量
            passes: LDA训练迭代次数
            minimum_probability: 主题概率的最小阈值
        """
        self.num_topics = num_topics
        self.passes = passes
        self.minimum_probability = minimum_probability
        self.dictionary = None
        self.lda_model = None
        
        # 添加默认停用词列表
        self.stop_words = set([
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
            '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个',
            '那个', '这些', '那些', '不是', '这是', '那是', '什么', '哪些',
            '这样', '那样', '之', '的话', '说', '对于', '等', '等等', '啊',
            '呢', '吧', '么', '哦', '嗯', '这', '那', '也', '还', '但',
            '但是', '不过', '然后', '因为', '所以', '如果', '虽然', '于是',
            '由于', '只是', '只有', '之一', '非常', '可以', '一些', '能够',
            '一样', '一直', '已经', '曾经', '正在', '将要', '可能', '应该',
            '自己','就是','还是','收起','时候','知道','adj','认为','觉得',
            '很多','这种','真的','现在','发现','怎么','如此','事情','东西','其实','比如'
        ])
        
        # 设置日志
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
    def add_stop_words(self, stop_words: List[str]):
        """
        添加自定义停用词
        
        Args:
            stop_words: 停用词列表
        """
        self.stop_words.update(stop_words)
    
    def load_stop_words_from_file(self, file_path: str):
        """
        从文件加载停用词
        
        Args:
            file_path: 停用词文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                stop_words = [line.strip() for line in f.readlines()]
            self.stop_words.update(stop_words)
            print(f"成功从文件加载 {len(stop_words)} 个停用词")
        except Exception as e:
            print(f"加载停用词文件失败：{str(e)}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        对中文文本进行预处理和分词
        
        Args:
            text: 输入的中文文本
            
        Returns:
            分词后的词语列表
        """
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 过滤停用词和空格
        return [word for word in words if word.strip() and len(word) > 1 and word not in self.stop_words]
    
    def fit(self, texts: List[str]):
        """
        训练LDA模型
        
        Args:
            texts: 文本列表
        """
        # 对所有文本进行分词
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 创建词典
        self.dictionary = corpora.Dictionary(processed_texts)
        
        # 创建文档-词语矩阵
        corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        # 训练LDA模型
        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            minimum_probability=self.minimum_probability,
            random_state=42
        )
        
    def analyze_text(self, text: str) -> List[tuple]:
        """
        分析单个文本的主题分布
        
        Args:
            text: 输入文本
            
        Returns:
            主题分布列表，每个元素为(主题ID, 概率)的元组
        """
        if not self.lda_model or not self.dictionary:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        processed_text = self.preprocess_text(text)
        bow = self.dictionary.doc2bow(processed_text)
        return self.lda_model.get_document_topics(bow)
    
    def print_topics(self, num_words: int = 10):
        """
        打印所有主题的关键词
        
        Args:
            num_words: 每个主题显示的关键词数量
        """
        if not self.lda_model:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        for idx, topic in self.lda_model.print_topics(num_words=num_words):
            print(f'主题 {idx + 1}:')
            print(topic)
            print()

    def evaluate_topics(self, texts: List[str], start_topics: int = 2, end_topics: int = 20, step: int = 1) -> int:
        """
        评估不同主题数量的效果
        
        Args:
            texts: 文本列表
            start_topics: 起始主题数量
            end_topics: 结束主题数量
            step: 步长
            
        Returns:
            最佳主题数量
        """
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # 存储不同主题数量的一致性得分
        coherence_scores = []
        
        print("\n开始评估不同主题数量的效果...")
        for num_topics in range(start_topics, end_topics + 1, step):
            print(f"\n评估主题数量 {num_topics}...")
            # 训练LDA模型
            lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=self.passes,
                random_state=42
            )
            
            # 计算一致性得分
            coherence_model = models.CoherenceModel(
                model=lda,
                texts=processed_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            print(f'主题数量 {num_topics}: 一致性得分 = {coherence_score:.4f}')
        
        # 找出最佳主题数量
        best_num_topics = range(start_topics, end_topics + 1, step)[
            coherence_scores.index(max(coherence_scores))
        ]
        print(f'\n最佳主题数量: {best_num_topics} (一致性得分: {max(coherence_scores):.4f})')
        return best_num_topics

def main():
    try:
        # 从Excel文件读取数据
        input_file = './nrfx/主题9.xlsx'
        output_file = './nrfx/主题9_topic_analysis_results.xlsx'
        
        df = pd.read_excel(input_file)
        texts = df['Title'].tolist()
        
        if not texts:
            raise ValueError("未找到数据，请确保Excel文件中'title'列包含文本")
            
        print(f"成功读取 {len(texts)} 条文本数据")
        
        # 首先评估最佳主题数量
        analyzer = ChineseTopicAnalyzer(passes=10)
        best_topics = analyzer.evaluate_topics(
            texts,
            start_topics=5,
            end_topics=9,
            step=2
        )
        
        # 使用评估得到的最佳主题数量
        analyzer = ChineseTopicAnalyzer(num_topics=best_topics, passes=10)
        
        # 训练模型
        analyzer.fit(texts)
        
        # 获取所有文本的主题分布
        all_topics = []
        for text in texts:
            topics = analyzer.analyze_text(text)
            # 只保留概率最高的主题
            max_topic = max(topics, key=lambda x: x[1])
            topic_dict = {
                "主要主题": f"主题{max_topic[0]+1}",
                "主题概率": max_topic[1]
            }
            all_topics.append(topic_dict)
        
        # 将主题分布添加到原始数据框
        topics_df = pd.DataFrame(all_topics)
        result_df = pd.concat([df, topics_df], axis=1)
        
        # 创建一个新的Excel写入器
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 第一个sheet：原始数据+主题分布
            result_df.to_excel(writer, sheet_name='主题分布', index=False)
            
            # 第二个sheet：主题关键词
            topic_keywords = []
            for topic_id in range(analyzer.num_topics):
                # 获取每个主题的关键词
                topic = analyzer.lda_model.show_topic(topic_id, topn=10)
                keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in topic])
                topic_keywords.append({
                    '主题ID': f'主题{topic_id+1}',
                    '关键词': keywords
                })
            
            pd.DataFrame(topic_keywords).to_excel(writer, sheet_name='主题关键词', index=False)
        
        print(f"\n分析结果已保存到文件：{output_file}")
        print("Sheet1 - 主题分布：包含原始数据和每条文本的主题分布")
        print("Sheet2 - 主题关键词：包含每个主题的关键词列表")
        
    except FileNotFoundError:
        print("未找到Excel文件，请检查文件路径是否正确")
    except KeyError:
        print("Excel文件中未找到'title'列，请检查列名是否正确")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main() 