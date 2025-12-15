import time
import tracemalloc
import cProfile
import pstats
import os
from bpetraining import train_bpe

def main():
    print("=== 问题4(a): BPE训练 ===")
    
    # 训练参数 - 文件在当前目录
    input_file = "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 5000
    special_tokens = ["<|endoftext|>"]
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        print("当前目录:", os.getcwd())
        print("目录内容:", os.listdir("."))
        return
    
    file_size = os.path.getsize(input_file) / (1024*1024)
    print(f"使用文件: {input_file}")
    print(f"文件大小: {file_size:.1f} MB")
    print(f"目标词汇表: {vocab_size}")
    print(f"特殊token: {special_tokens}")
    
    # 1. 带监控的训练
    print("\n开始训练...")
    tracemalloc.start()
    start_time = time.time()
    
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
    
    training_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 2. 分析结果
    print("\n=== 训练完成! ===")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"峰值内存使用: {peak_mem / 1024 / 1024:.2f} MB")
    print(f"词汇表大小: {len(vocab)}")
    print(f"合并次数: {len(merges)}")
    
    # 找最长token
    max_len = 0
    longest_tokens = []
    for token_id, token_bytes in vocab.items():
        if token_id >= 257:  # 跳过单字节和特殊token
            token_len = len(token_bytes)
            if token_len > max_len:
                max_len = token_len
                longest_tokens = [(token_id, token_bytes)]
            elif token_len == max_len:
                longest_tokens.append((token_id, token_bytes))
    
    print(f"\n最长token长度: {max_len} 字节")
    if longest_tokens:
        token_id, token_bytes = longest_tokens[0]
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
            print(f"示例最长token (ID={token_id}): {repr(token_str)}")
            print(f"十六进制: {token_bytes.hex()}")
        except:
            print(f"示例最长token (ID={token_id}): 无法解码为UTF-8")
            print(f"十六进制: {token_bytes.hex()}")
    
    # 3. 性能分析
    print("\n=== 问题4(b): 性能分析 ===")
    print("运行性能分析...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 重新运行一次用于分析
    vocab2, merges2 = train_bpe(input_file, vocab_size, special_tokens)
    
    profiler.disable()
    
    # 输出分析结果
    stats = pstats.Stats(profiler)
    print("\n按累计时间排序的前10个函数:")
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)
    
    print("\n按内部时间排序的前10个函数:")
    stats.sort_stats('time').print_stats(10)
    
    # 保存分析结果
    profiler.dump_stats("bpe_performance.prof")
    print(f"\n详细性能分析已保存到: bpe_performance.prof")
    print("使用 'snakeviz bpe_performance.prof' 可视化查看")
    
    # 生成答案文件
    with open("problem4_answer.txt", "w", encoding="utf-8") as f:
        f.write("# 问题4答案\n\n")
        f.write("## (a) BPE训练结果\n")
        f.write(f"- 训练文件: TinyStoriesV2-GPT4-valid.txt ({file_size:.1f}MB)\n")
        f.write(f"- 目标vocab_size: {vocab_size}\n")
        f.write(f"- 实际vocab_size: {len(vocab)}\n")
        f.write(f"- 训练时间: {training_time:.2f} 秒\n")
        f.write(f"- 峰值内存使用: {peak_mem / 1024 / 1024:.2f} MB\n")
        f.write(f"- 最长token长度: {max_len} 字节\n")
        if longest_tokens:
            token_id, token_bytes = longest_tokens[0]
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
                f.write(f"- 最长token示例: {repr(token_str)}\n")
            except:
                f.write(f"- 最长token示例: (无法解码为UTF-8)\n")
        
        f.write("\n## (b) 性能分析\n")
        f.write("根据profiler输出，最耗时的部分通常是:\n")
        f.write("1. 主循环中的合并操作 (while current_vocab_size < vocab_size and pair_counts:)\n")
        f.write("2. 查找最佳pair的循环 (for pair, cnt in pair_counts.items():)\n")
        f.write("3. 更新pair_counts和pair_to_word_indices字典\n")
        f.write("4. 文本预处理 (_pretokenize_chunk 函数)\n")
        
    print(f"\n答案已保存到: problem4_answer.txt")

if __name__ == "__main__":
    main()
