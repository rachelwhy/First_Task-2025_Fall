import re
import regex
from typing import Dict, List, Tuple, Any, Union
import json
import pickle
import os


class OptimizedBPE:
    def __init__(self, vocab_size: int = 1000,
                 special_tokens: List[str] = None,
                 lowercase: bool = False):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.special_tokens = special_tokens or []
        self.gpt2_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.vocab: Dict[int, bytes] = {}  # id -> bytes
        self.merges: List[Tuple[bytes, bytes]] = []
        self.id_to_token: Dict[int, bytes] = {}
        self.token_to_id: Dict[bytes, int] = {}
        self.str_vocab: Dict[str, int] = {}
        self.id_to_str_token: Dict[int, str] = {}
        self.special_token_bytes: Dict[str, bytes] = {}
        self.special_token_ids: Dict[str, int] = {}
        self._cache: Dict[str, List[str]] = {}

    def initialize_from_existing(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.str_vocab = {}
        self.id_to_str_token = {}
        for token_id, token_bytes in vocab.items():
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_str = token_bytes.hex()
            self.str_vocab[token_str] = token_id
            self.id_to_str_token[token_id] = token_str

        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            self.special_token_bytes[token] = token_bytes

            if token_bytes in self.token_to_id:
                self.special_token_ids[token] = self.token_to_id[token_bytes]
            else:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = token_bytes
                self.id_to_token[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
                self.special_token_ids[token] = new_id
                self.str_vocab[token] = new_id
                self.id_to_str_token[new_id] = token

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_special_tokens]
            special_pattern = re.compile(f"({'|'.join(escaped)})")
            parts = special_pattern.split(text)
        else:
            parts = [text]

        all_ids = []
        for part in parts:
            if not part:
                continue

            if part in self.special_token_ids:
                all_ids.append(self.special_token_ids[part])
            else:
                pretokens = self.gpt2_pattern.findall(part)
                for pretoken in pretokens:
                    ids = self._apply_bpe_to_pretoken(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def tokenize(self, text: str) -> List[str]:
        ids = self.encode(text)

        tokens = []
        for token_id in ids:
            if token_id in self.id_to_str_token:
                tokens.append(self.id_to_str_token[token_id])
            elif token_id in self.id_to_token:
                token_bytes = self.id_to_token[token_id]
                try:
                    tokens.append(token_bytes.decode('utf-8'))
                except UnicodeDecodeError:
                    tokens.append(token_bytes.hex())
            else:
                tokens.append(f"[UNK:{token_id}]")

        return tokens

    def _apply_bpe_to_pretoken(self, pretoken: str) -> List[int]:
        if not pretoken:
            return []

        pretoken_bytes = pretoken.encode('utf-8')

        tokens = [bytes([b]) for b in pretoken_bytes]

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

        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])

        return ids

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""

        flat_ids = []
        for item in ids:
            if isinstance(item, list):
                flat_ids.extend(item)
            else:
                flat_ids.append(item)

        result_bytes = b''
        for token_id in flat_ids:
            if token_id in self.id_to_token:
                result_bytes += self.id_to_token[token_id]

        try:
            return result_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return result_bytes.decode('utf-8', errors='replace')

    def encode_iterable(self, texts: Any) -> List[List[int]]:
        if hasattr(texts, 'read'):
            content = texts.read()
            lines = content.splitlines(keepends=False)
            text_list = [line for line in lines if line]
        elif isinstance(texts, list):
            text_list = texts
        else:
            text_list = list(texts)

        all_ids = []
        for text in text_list:
            if text:
                ids = self.encode(text)
                all_ids.extend(ids)

        return [all_ids]

    def save(self, path: str):
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'OptimizedBPE':
        with open(path, 'rb') as f:
            data = pickle.load(f)

        bpe = cls(
            vocab_size=len(data['vocab']),
            special_tokens=data['special_tokens']
        )
        bpe.initialize_from_existing(data['vocab'], data['merges'])
        return bpe


def test_gpt2_compatibility():
    vocab = {
        0: b'<|endoftext|>',
        82: b's',
        198: b'Hello',
        2202: b'He',
        344: b'llo',
        4776: b',',
        612: b' how',
        3932: b' are',
        50256: b'<|endoftext|>',
    }

    merges = [
        (b'H', b'e'),   # He
        (b'e', b'l'),   # el
        (b'l', b'l'),   # ll
        (b'll', b'o'),  # llo
    ]

    bpe = OptimizedBPE(
        vocab_size=len(vocab),
        special_tokens=['<|endoftext|>']
    )
    bpe.initialize_from_existing(vocab, merges)

    test_text = "Hello, how are you?"
    tokens = bpe.tokenize(test_text)
    print(f"text: {test_text}")
    print(f"bpe: {tokens}")

    ids = bpe.encode(test_text)
    print(f"code: {ids}")
    print(f"decode: {bpe.decode(ids)}")


if __name__ == "__main__":
    test_gpt2_compatibility()