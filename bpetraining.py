"""
BPE training implementation.

Further-optimized version:
- Builds initial pair -> total count and pair -> set(word_indices)
- On each merge, only touch words that contain the chosen pair:
  * remove their old pair contributions
  * apply the merge to that word
  * add new pair contributions for that word
- Avoids rebuilding global pair counts from scratch each iteration.
- Keeps same API as original: train_bpe(input_path, vocab_size, special_tokens)
"""

from collections import Counter, defaultdict
from pathlib import Path
import regex as re
from typing import Dict, List, Tuple

# GPT-2 regex pattern (as required by the spec)
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

_token_re = re.compile(PAT)

def _split_on_specials(text: str, special_tokens: List[str]) -> List[str]:
    """Split text into chunks that do not cross special token boundaries.
    The special tokens themselves are kept as chunks."""
    if not special_tokens:
        return [text]
    esc = [re.escape(st) for st in special_tokens]
    splitter = re.compile("(" + "|".join(esc) + ")")
    parts = splitter.split(text)
    return [p for p in parts if p != ""]

def _pretokenize_chunk(chunk: str, special_tokens_set: set) -> List[Tuple[bytes, ...]]:
    """Pre-tokenize a chunk. Protected special tokens are returned as single tokens.
    Otherwise apply GPT-2 regex and return tuple-of-single-byte-bytes symbols for each token."""
    if chunk in special_tokens_set:
        return [(chunk.encode("utf-8"),)]

    out = []
    for m in _token_re.finditer(chunk):
        tok = m.group(0)
        if not tok:
            continue
        b = tok.encode("utf-8")
        symbols = tuple(bytes([bb]) for bb in b)
        if symbols:
            out.append(symbols)
    return out

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    """Train BPE tokenizer.

    Returns:
      vocab: dict[int, bytes]  -- mapping token id -> byte sequence (as bytes)
      merges: list[tuple[bytes, bytes]] -- list of merges in creation order
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    text = p.read_text(encoding="utf-8")

    # split on special tokens first (they are protected)
    chunks = _split_on_specials(text, special_tokens)
    special_set = set(special_tokens)

    # pretokenize all chunks and count frequencies of "words" (tuples of symbols)
    word_freq: Counter = Counter()
    for chunk in chunks:
        tokens = _pretokenize_chunk(chunk, special_set)
        for t in tokens:
            word_freq[t] += 1

    # Initialize vocab list (bytes sequences)
    vocab_list: List[bytes] = []
    # Add special tokens first as full-byte sequences
    for st in special_tokens:
        vocab_list.append(st.encode("utf-8"))
    # Add 256 single-byte tokens
    for i in range(256):
        vocab_list.append(bytes([i]))

    merges: List[Tuple[bytes, bytes]] = []

    # Convert word_freq to mutable list-of-lists + freqs
    words: List[List[bytes]] = [list(w) for w in word_freq.keys()]  # each w: list of symbols (bytes)
    freqs: List[int] = list(word_freq.values())
    n_words = len(words)

    # Build initial pair_counts and mapping pair -> set(word_indices)
    pair_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_word_indices: Dict[Tuple[bytes, bytes], set] = defaultdict(set)

    for idx, (w, f) in enumerate(zip(words, freqs)):
        ln = len(w)
        for i in range(ln - 1):
            pair = (w[i], w[i + 1])
            pair_counts[pair] += f
            pair_to_word_indices[pair].add(idx)

    current_vocab_size = len(vocab_list)

    # Main loop: merge until target vocab_size
    while current_vocab_size < vocab_size and pair_counts:
        # Find best pair: highest freq, tie-break by lexicographic greater pair
        # Note: iterating over dict items is fine; number of distinct pairs is typically modest
        best_pair = None
        best_freq = -1
        for pair, cnt in pair_counts.items():
            if cnt > best_freq or (cnt == best_freq and (best_pair is None or pair > best_pair)):
                best_pair = pair
                best_freq = cnt

        if best_pair is None or best_freq <= 0:
            break

        a, b = best_pair
        new_symbol = a + b

        # Record merge and add to vocab
        merges.append((a, b))
        vocab_list.append(new_symbol)
        current_vocab_size += 1

        # Get affected word indices (copy because we'll modify the sets)
        affected_indices = list(pair_to_word_indices.get(best_pair, set()))
        if not affected_indices:
            # no words actually contain it any more (defensive)
            # remove pair and continue
            pair_counts.pop(best_pair, None)
            pair_to_word_indices.pop(best_pair, None)
            continue

        # For each affected word: remove old pair contributions, modify word, then add new pair contributions
        for idx in affected_indices:
            # If this word was removed/empty somehow, skip
            if idx >= len(words):
                continue
            w = words[idx]
            f = freqs[idx]
            if len(w) < 2:
                # nothing to do
                continue

            # Compute old pairs for this word (list)
            old_pairs = []
            for i in range(len(w) - 1):
                old_pairs.append((w[i], w[i + 1]))

            # Remove this word's contribution from global pair_counts and pair_to_word_indices
            # (for each old pair, decrement and remove idx from set)
            for pair in old_pairs:
                # decrement count
                cnt = pair_counts.get(pair, 0)
                if cnt <= f:
                    # remove entirely
                    pair_counts.pop(pair, None)
                else:
                    pair_counts[pair] = cnt - f
                # remove index from mapping set
                s = pair_to_word_indices.get(pair)
                if s:
                    s.discard(idx)
                    if len(s) == 0:
                        pair_to_word_indices.pop(pair, None)

            # Apply merge on this word: replace adjacent (a,b) with new_symbol
            new_w = []
            i = 0
            changed = False
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new_w.append(new_symbol)
                    i += 2
                    changed = True
                else:
                    new_w.append(w[i])
                    i += 1

            # If no change (maybe pair no longer present due to earlier merges), skip adding contributions
            words[idx] = new_w

            if not changed:
                continue

            # Compute new pairs for this modified word and add contributions
            ln2 = len(new_w)
            for j in range(ln2 - 1):
                p = (new_w[j], new_w[j + 1])
                pair_counts[p] += f
                pair_to_word_indices[p].add(idx)

        # Finally, remove the merged pair from maps if present (it's now obsolete)
        pair_counts.pop(best_pair, None)
        pair_to_word_indices.pop(best_pair, None)

    # Build final vocab mapping id -> bytes
    vocab: Dict[int, bytes] = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merges