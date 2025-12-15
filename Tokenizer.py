"""优化的BPE实现 - 针对GPT-2 tokenizer兼容性修复"""

import re
import regex  # 需要安装: pip install regex
from typing import Dict, List, Tuple, Any, Union
import json
import pickle
import os


class OptimizedBPE:
    def __init__(self, vocab_size: int = 1000,
                 special_tokens: List[str] = None,
                 lowercase: bool = False):
        """
        专门为GPT-2兼容性设计的BPE tokenizer
        """
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.special_tokens = special_tokens or []

        # GPT-2的正则表达式模式
        self.gpt2_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # 词汇表结构
        self.vocab: Dict[int, bytes] = {}  # id -> bytes
        self.merges: List[Tuple[bytes, bytes]] = []  # 合并规则列表

        # 反向映射
        self.id_to_token: Dict[int, bytes] = {}
        self.token_to_id: Dict[bytes, int] = {}

        # 字符串表示的词汇表（用于tokenize方法）
        self.str_vocab: Dict[str, int] = {}
        self.id_to_str_token: Dict[int, str] = {}

        # 特殊token处理
        self.special_token_bytes: Dict[str, bytes] = {}
        self.special_token_ids: Dict[str, int] = {}

        # 缓存
        self._cache: Dict[str, List[str]] = {}

    def initialize_from_existing(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]):
        """从已有的词汇表和合并规则初始化"""
        self.vocab = vocab
        self.merges = merges

        # 构建反向映射
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}

        # 构建字符串表示的词汇表（用于tokenize方法）
        self.str_vocab = {}
        self.id_to_str_token = {}
        for token_id, token_bytes in vocab.items():
            # 将bytes转换为字符串表示
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # 对于无效UTF-8，使用hex表示
                token_str = token_bytes.hex()
            self.str_vocab[token_str] = token_id
            self.id_to_str_token[token_id] = token_str

        # 处理特殊token
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            self.special_token_bytes[token] = token_bytes

            if token_bytes in self.token_to_id:
                self.special_token_ids[token] = self.token_to_id[token_bytes]
            else:
                # 添加到词汇表末尾
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = token_bytes
                self.id_to_token[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
                self.special_token_ids[token] = new_id

                # 同时添加到字符串词汇表
                self.str_vocab[token] = new_id
                self.id_to_str_token[new_id] = token

    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs - 使用GPT-2的预分词方法"""
        if not text:
            return []

        # 处理特殊token分割
        if self.special_tokens:
            # 按长度降序排序
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_special_tokens]
            special_pattern = re.compile(f"({'|'.join(escaped)})")
            parts = special_pattern.split(text)
        else:
            parts = [text]

        # 对每个部分进行编码
        all_ids = []
        for part in parts:
            if not part:
                continue

            # 检查是否是特殊token
            if part in self.special_token_ids:
                all_ids.append(self.special_token_ids[part])
            else:
                # 使用GPT-2正则进行预分词
                pretokens = self.gpt2_pattern.findall(part)
                for pretoken in pretokens:
                    ids = self._apply_bpe_to_pretoken(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def tokenize(self, text: str) -> List[str]:
        """对文本进行分词，返回token字符串列表"""
        # 首先编码为IDs
        ids = self.encode(text)

        # 然后将IDs转换为token字符串
        tokens = []
        for token_id in ids:
            if token_id in self.id_to_str_token:
                tokens.append(self.id_to_str_token[token_id])
            elif token_id in self.id_to_token:
                # 将bytes转换为字符串
                token_bytes = self.id_to_token[token_id]
                try:
                    tokens.append(token_bytes.decode('utf-8'))
                except UnicodeDecodeError:
                    tokens.append(token_bytes.hex())
            else:
                # 未知token，使用占位符
                tokens.append(f"[UNK:{token_id}]")

        return tokens

    def _apply_bpe_to_pretoken(self, pretoken: str) -> List[int]:
        """对预分词结果应用BPE"""
        if not pretoken:
            return []

        # 转换为字节
        pretoken_bytes = pretoken.encode('utf-8')

        # 初始化为字节
        tokens = [bytes([b]) for b in pretoken_bytes]

        # 按顺序应用所有合并规则
        for a, b in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 转换为IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])

        return ids

    def decode(self, ids: List[int]) -> str:
        """解码token IDs为文本"""
        if not ids:
            return ""

        # 展平列表
        flat_ids = []
        for item in ids:
            if isinstance(item, list):
                flat_ids.extend(item)
            else:
                flat_ids.append(item)

        # 转换为字节
        result_bytes = b''
        for token_id in flat_ids:
            if token_id in self.id_to_token:
                result_bytes += self.id_to_token[token_id]

        # 解码为字符串
        try:
            return result_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return result_bytes.decode('utf-8', errors='replace')

    def encode_iterable(self, texts: Any) -> List[List[int]]:
        """批量编码"""
        # 读取所有文本
        if hasattr(texts, 'read'):
            content = texts.read()
            lines = content.splitlines(keepends=False)
            text_list = [line for line in lines if line]
        elif isinstance(texts, list):
            text_list = texts
        else:
            text_list = list(texts)

        # 编码所有文本
        all_ids = []
        for text in text_list:
            if text:
                ids = self.encode(text)
                all_ids.extend(ids)

        # 返回嵌套列表格式
        return [all_ids]

    def save(self, path: str):
        """保存模型"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'OptimizedBPE':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        bpe = cls(
            vocab_size=len(data['vocab']),
            special_tokens=data['special_tokens']
        )
        bpe.initialize_from_existing(data['vocab'], data['merges'])
        return bpe


# 测试函数
def test_gpt2_compatibility():
    """测试GPT-2兼容性"""
    # 创建一个简单的词汇表用于测试
    vocab = {
        0: b'<|endoftext|>',
        82: b's',
        198: b'Hello',
        2202: b'He',
        344: b'llo',
        4776: b',',
        612: b' how',
        3932: b' are',
        50256: b'<|endoftext|>',  # 特殊token ID
    }

    merges = [
        (b'H', b'e'),   # He
        (b'e', b'l'),   # el (可能)
        (b'l', b'l'),   # ll
        (b'll', b'o'),  # llo
    ]

    # 创建tokenizer
    bpe = OptimizedBPE(
        vocab_size=len(vocab),
        special_tokens=['<|endoftext|>']
    )
    bpe.initialize_from_existing(vocab, merges)

    # 测试tokenize方法
    test_text = "Hello, how are you?"
    tokens = bpe.tokenize(test_text)
    print(f"文本: {test_text}")
    print(f"分词: {tokens}")

    ids = bpe.encode(test_text)
    print(f"编码: {ids}")
    print(f"解码: {bpe.decode(ids)}")


if __name__ == "__main__":
    test_gpt2_compatibility()