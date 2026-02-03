iimport time
import tracemalloc
import cProfile
import pstats
import os
from bpetraining import train_bpe

def main():
    print("=== Question 4(a): BPE Training ===")

    # Training parameters - file is in current directory
    input_file = "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 5000
    special_tokens = ["<|endoftext|>"]

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File does not exist: {input_file}")
        print("Current directory:", os.getcwd())
        print("Directory contents:", os.listdir("."))
        return

    file_size = os.path.getsize(input_file) / (1024*1024)
    print(f"Using file: {input_file}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Target vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # 1. Training with monitoring
    print("\nStarting training...")
    tracemalloc.start()
    start_time = time.time()

    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)

    training_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Result analysis
    print("\n=== Training completed! ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Peak memory usage: {peak_mem / 1024 / 1024:.2f} MB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Find longest token
    max_len = 0
    longest_tokens = []
    for token_id, token_bytes in vocab.items():
        if token_id >= 257:  # Skip single bytes and special tokens
            token_len = len(token_bytes)
            if token_len > max_len:
                max_len = token_len
                longest_tokens = [(token_id, token_bytes)]
            elif token_len == max_len:
                longest_tokens.append((token_id, token_bytes))

    print(f"\nLongest token length: {max_len} bytes")
    if longest_tokens:
        token_id, token_bytes = longest_tokens[0]
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
            print(f"Example longest token (ID={token_id}): {repr(token_str)}")
            print(f"Hexadecimal: {token_bytes.hex()}")
        except:
            print(f"Example longest token (ID={token_id}): Cannot decode as UTF-8")
            print(f"Hexadecimal: {token_bytes.hex()}")

    # 3. Performance analysis
    print("\n=== Question 4(b): Performance Analysis ===")
    print("Running performance analysis...")

    profiler = cProfile.Profile()
    profiler.enable()

    # Run again for profiling
    vocab2, merges2 = train_bpe(input_file, vocab_size, special_tokens)

    profiler.disable()

    # Output analysis results
    stats = pstats.Stats(profiler)
    print("\nTop 10 functions by cumulative time:")
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)

    print("\nTop 10 functions by internal time:")
    stats.sort_stats('time').print_stats(10)

    # Save analysis results
    profiler.dump_stats("bpe_performance.prof")
    print(f"\nDetailed performance analysis saved to: bpe_performance.prof")
    print("Use 'snakeviz bpe_performance.prof' for visualization")

    # Generate answer file
    with open("problem4_answer.txt", "w", encoding="utf-8") as f:
        f.write("# Question 4 Answer\n\n")
        f.write("## (a) BPE Training Results\n")
        f.write(f"- Training file: TinyStoriesV2-GPT4-valid.txt ({file_size:.1f}MB)\n")
        f.write(f"- Target vocab_size: {vocab_size}\n")
        f.write(f"- Actual vocab_size: {len(vocab)}\n")
        f.write(f"- Training time: {training_time:.2f} seconds\n")
        f.write(f"- Peak memory usage: {peak_mem / 1024 / 1024:.2f} MB\n")
        f.write(f"- Longest token length: {max_len} bytes\n")
        if longest_tokens:
            token_id, token_bytes = longest_tokens[0]
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
                f.write(f"- Longest token example: {repr(token_str)}\n")
            except:
                f.write(f"- Longest token example: (Cannot decode as UTF-8)\n")

        f.write("\n## (b) Performance Analysis\n")
        f.write("According to profiler output, the most time-consuming parts are usually:\n")
        f.write("1. Merge operations in the main loop (while current_vocab_size < vocab_size and pair_counts:)\n")
        f.write("2. Loop for finding the best pair (for pair, cnt in pair_counts.items():)\n")
        f.write("3. Updating pair_counts and pair_to_word_indices dictionaries\n")
        f.write("4. Text preprocessing (_pretokenize_chunk function)\n")

    print(f"\nAnswer saved to: problem4_answer.txt")

if __name__ == "__main__":
    main()