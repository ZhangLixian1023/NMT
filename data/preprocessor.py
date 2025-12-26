import re
import jsonlines
import hanlp
import nltk
from typing import List, Tuple, Dict


class Preprocessor:
    def __init__(self):
        # 初始化HanLP分词器
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

    def clean_text(self, text: str, language: str) -> str:
        """
        文本清洗：保留有效字符，移除噪声
        注意：不移除标点（nltk/jieba 需要它们进行正确分词）
        """

        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)  # 移除控制字符（必须）
        if language == 'en':
            text = text.lower()

        if language == 'zh':
            # 中文仍需过滤（因分词器对噪声敏感）
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:\'"()\-\[\]/–—\u3000-\u303F\uFF01-\uFF5E]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str, language: str) -> str:
        """
        分词函数
        :return: 空格分隔的 token 字符串（用于后续 build_vocab）
        """

        if language == 'en':
            # 使用 nltk.word_tokenize（能正确处理 "don't", "U.S.", "1989," 等）
            tokens = nltk.word_tokenize(text)
            return ' '.join(tokens)
        else:  # zh
            # 使用HanLP分词器
            tokens = self.tokenizer(text)
            return ' '.join(tokens)

    def process_text(self, text: str, language: str) -> str:
        """完整处理流程"""
        text = self.clean_text(text, language)
        text = self.tokenize(text, language)
        return text

    def load_data(self, file_paths: List[str]) -> List[Dict[str, str]]:
        """加载 jsonl 数据"""
        data = []
        for file_path in file_paths:
            with jsonlines.open(file_path, 'r') as reader:
                for item in reader:
                    data.append(item)
        return data


    def prepare_data(self, data: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """生成 (en, zh) 平行语料"""
        processed_data = []
        for item in data:
            en_text = self.process_text(item['en'], 'en')
            zh_text = self.process_text(item['zh'], 'zh')
            if en_text and zh_text:  # 过滤空句子
                processed_data.append((zh_text,en_text))
        return processed_data