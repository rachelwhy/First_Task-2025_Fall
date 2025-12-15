from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Optional, List, Tuple
# 【关键修复】使用 regex 库来支持 GPT-2 正则表达式中的 \p{L}+ 和 \p{N}+
import regex as re
from bpetraining import train_bpe  # 导入 train_bpe

# 尝试导入你的 OptimizedBPE
try:
    # 假设 Tokenizer.py 中存在 OptimizedBPE 类
    from Tokenizer import OptimizedBPE

    OPTIMIZED_BPE_AVAILABLE = True
except ImportError:
    OPTIMIZED_BPE_AVAILABLE = False

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# GPT-2 regex pattern (从 bpetraining.py 复制)
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_token_re = re.compile(PAT)


def run_linear(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.
    """
    raise NotImplementedError


def run_embedding(
        vocab_size: int,
        d_model: int,
        weights: Float[Tensor, " vocab_size d_model"],
        token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """
    raise NotImplementedError


def run_swiglu(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.
    """
    raise NotImplementedError


def run_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.
    """
    raise NotImplementedError


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This version of MHA should include RoPE.
    """
    raise NotImplementedError


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.
    """
    raise NotImplementedError


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.
    """
    raise NotImplementedError


def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.
    """
    raise NotImplementedError


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, " d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.
    """
    raise NotImplementedError


def run_get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.
    """
    raise NotImplementedError


def run_cross_entropy(
        inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """
    raise NotImplementedError


def run_load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.
    """
    raise NotImplementedError


def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    # 尝试使用 OptimizedBPE 实现 (如果可用)
    if 'OPTIMIZED_BPE_AVAILABLE' in globals() and OPTIMIZED_BPE_AVAILABLE:
        try:
            # 导入 OptimizedBPETokenizer (假设它在 Tokenizer.py 中)
            from Tokenizer import OptimizedBPETokenizer
            return OptimizedBPETokenizer(vocab, merges, special_tokens)
        except ImportError:
            pass

            # =======================================================

    # 核心 BPETokenizer 实现 (回退/默认实现)
    # =======================================================

    class BPETokenizer:
        def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                     special_tokens: Optional[list[str]] = None):
            self.vocab = vocab
            self.merges = merges
            self.special_tokens = special_tokens or []

            # 创建反向映射 (核心)
            self.id_to_token = vocab.copy()
            self.token_to_id = {v: k for k, v in vocab.items()}

            # 构建合并映射（按优先级排序）
            self.merge_dict = {pair: i for i, pair in enumerate(merges)}

            # 特殊token处理
            if self.special_tokens:
                # 按长度降序排序，确保长token优先匹配，并转义
                sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
                escaped_tokens = [re.escape(token) for token in sorted_special_tokens]

                # 构建特殊token的分隔/匹配模式
                self.special_token_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

                # 填充特殊token ID和bytes
                for token in self.special_tokens:
                    token_bytes = token.encode('utf-8')
                    if token_bytes not in self.token_to_id:
                        # 找到下一个可用的ID
                        new_id = max(self.id_to_token.keys()) + 1 if self.id_to_token else 0
                        self.id_to_token[new_id] = token_bytes
                        self.token_to_id[token_bytes] = new_id

                # 重新映射特殊token ID (用于快速查找)
                self.special_token_ids = {
                    token: self.token_to_id[token.encode('utf-8')]
                    for token in self.special_tokens
                }
            else:
                self.special_token_pattern = None
                self.special_token_ids = {}

        def _bpe_tokenize_block(self, block_bytes: bytes) -> List[bytes]:
            """对单个预分词块的字节序列应用BPE合并"""
            if not block_bytes:
                return []

            # 初始化为单个字节的tokens
            tokens = [bytes([b]) for b in block_bytes]

            # 持续合并，直到没有可应用的合并
            while True:
                # 找到最优先级的合并对 (最低索引)
                best_pair = None
                best_priority = float('inf')

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_dict:
                        priority = self.merge_dict[pair]
                        if priority < best_priority:
                            best_priority = priority
                            best_pair = pair

                if best_pair is None:
                    # 没有可用的合并，退出循环
                    break

                # 执行合并
                a, b = best_pair
                new_tokens = []
                i = 0
                while i < len(tokens):  # 遍历 tokens
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            return tokens

        def encode(self, text: str) -> List[int]:
            """将文本编码为token ids，正确处理特殊token和预分词"""
            if not text:
                return []

            result_ids = []

            # 1. 使用特殊token模式分割文本
            if self.special_token_pattern:
                parts = self.special_token_pattern.split(text)
            else:
                parts = [text]

            for part in parts:
                if not part:
                    continue

                if part in self.special_token_ids:
                    # 2. 特殊token：直接添加ID
                    result_ids.append(self.special_token_ids[part])
                else:
                    # 3. 普通文本：应用GPT-2预分词
                    pretokens = _token_re.findall(part)
                    for pretoken in pretokens:
                        if not pretoken:
                            continue

                        # 转换为字节
                        pretoken_bytes = pretoken.encode('utf-8')

                        # 4. 在每个预分词块内应用 BPE 合并
                        token_bytes_list = self._bpe_tokenize_block(pretoken_bytes)

                        # 5. 转换为 token IDs
                        for token_bytes in token_bytes_list:
                            if token_bytes in self.token_to_id:
                                result_ids.append(self.token_to_id[token_bytes])
                            else:
                                # 回退到字节级别 (BPE 的核心保证)
                                for byte in token_bytes:
                                    byte_token = bytes([byte])
                                    if byte_token in self.token_to_id:
                                        result_ids.append(self.token_to_id[byte_token])
            return result_ids

        def decode(self, token_ids: List[int]) -> str:
            """将token ids解码为文本"""
            if not token_ids:
                return ""

            # 展平嵌套列表（如果传入的是嵌套列表）
            flat_ids = []
            for item in token_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)

            # 将token ids转换为字节
            tokens_bytes = b''
            for token_id in flat_ids:
                if token_id in self.id_to_token:
                    tokens_bytes += self.id_to_token[token_id]

            # 将字节解码为文本，使用 errors='replace'
            return tokens_bytes.decode('utf-8', errors='replace')

        def encode_iterable(self, texts: Iterable[str]) -> List[int]:
            """批量编码文本，并返回展平的 ID 列表 (以匹配测试期望)"""
            all_ids = []

            if hasattr(texts, 'read'):  # 文件对象
                content = texts.read()
                lines = content.splitlines(keepends=True)
                for line in lines:
                    if line:
                        # 编码整个行，包括换行符（如果存在），以匹配文件的原始内容
                        encoded_ids = self.encode(line)
                        all_ids.extend(encoded_ids)
            else:
                # 普通迭代器
                for text in texts:
                    if hasattr(text, 'strip'):
                        text = text.strip()
                    if text:
                        all_ids.extend(self.encode(text))

            # 返回展平的 ID 列表
            return all_ids

    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """
    # 直接调用 bpetraining.py 中的 train_bpe 函数
    return train_bpe(input_path, vocab_size, special_tokens)