
from collections import Counter
import math
def calculate_bleu4(references, hypotheses):
    """
    根据给定的参考译文和假设译文列表，计算 BLEU-4 分数及其组成部分 p1, p2, p3, p4。
    不使用 brevity penalty。
    
    Args:
        references: List of lists. 每个元素是一个参考译文（词列表）。
        hypotheses: List of lists. 每个元素是模型生成的译文（词列表）。
    
    Returns:
        tuple: (bleu4_score, p1, p2, p3, p4)
    """
    # 初始化 n-gram 精确度的分子和分母
    numerator = [0, 0, 0, 0]  # p1, p2, p3, p4 的分子
    denominator = [0, 0, 0, 0]  # p1, p2, p3, p4 的分母

    # 遍历每一对参考译文和假设译文
    for i, (ref_list, hyp) in enumerate(zip(references, hypotheses)):
        ref = ref_list

        # 对于每个 n (1到4)
        for n in range(1, 5):
            # 计算假设译文的 n-gram
            hyp_ngrams = []
            for j in range(len(hyp) - n + 1):
                ngram = tuple(hyp[j:j+n])
                hyp_ngrams.append(ngram)
            
            # 计算参考译文的 n-gram 及其最大计数
            ref_ngrams_count = {}
            for j in range(len(ref) - n + 1):
                ngram = tuple(ref[j:j+n])
                ref_ngrams_count[ngram] = ref_ngrams_count.get(ngram, 0) + 1
            
            # 计算匹配的 n-gram 数量
            matched = 0
            hyp_ngrams_count = Counter(hyp_ngrams)
            for ngram, count in hyp_ngrams_count.items():
                if ngram in ref_ngrams_count:
                    matched += min(count, ref_ngrams_count[ngram])
            
            # 累加分子和分母
            numerator[n-1] += matched
            denominator[n-1] += len(hyp_ngrams)

    # 计算 p1, p2, p3, p4
    p1 = numerator[0] / denominator[0] if denominator[0] > 0 else 0
    p2 = numerator[1] / denominator[1] if denominator[1] > 0 else 0
    p3 = numerator[2] / denominator[2] if denominator[2] > 0 else 0
    p4 = numerator[3] / denominator[3] if denominator[3] > 0 else 0

    bleu4 = p1 * p2 * p3 * p4

    return bleu4, p1, p2, p3, p4