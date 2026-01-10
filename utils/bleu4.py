
from collections import Counter
import math 
def calculate_bleu4(hypotheses, references):
    """
    根据给定的参考译文和假设译文列表，计算 BLEU-4 分数及其组成部分 p1, p2, p3, p4
    
    Args:
        传入的数组是许多句话构成的，每句话是词的列表 
        references: List of lists. 每个元素是一句参考译文（词列表）。
        hypotheses: List of lists. 每个元素是模型生成的译文（词列表）。
    
    Returns:
        dict: {
        ppt_bleu_score,
        sacrebleu_score, 
        correct:list[4],
        total:list[4],
        percent:list[4],
        hyps_len,
        ref_len
        }
    """
    # 初始化 n-gram 精确度的分子和分母
    c = [0, 0, 0, 0]  # p1, p2, p3, p4 的分子
    t = [0, 0, 0, 0]  # p1, p2, p3, p4 的分母
    p = [0., 0., 0., 0.]  # p1, p2, p3, p4 的精确度 平滑的
    hyp_len = 0
    ref_len = 0

    #references=[[1,3,4,5,7]]
    #hypotheses=[[1,3,4,5,7]]

    # 遍历每一对参考译文和假设译文
    # 传入的数组是许多句话构成的，每句话是词的列表 
    for (ref, hyp) in zip(references, hypotheses):
        h=len(hyp)
        r=len(ref)
        hyp_len += h
        ref_len += r
        # 对于每个 n (1到4)
        for n in range(1, 5):
            # 计算假设译文的 n-gram
            hyp_ngrams = []
            for j in range(h - n + 1):
                ngram = tuple(hyp[j:j+n])
                # print(ngram)
                hyp_ngrams.append(ngram)
            
            # 计算参考译文的 n-gram 及其最大计数
            ref_ngrams_count = {}
            for j in range(r - n + 1):
                ngram = tuple(ref[j:j+n])
                ref_ngrams_count[ngram] = ref_ngrams_count.get(ngram, 0) + 1
            
            # 计算匹配的 n-gram 数量
            matched = 0
            hyp_ngrams_count = Counter(hyp_ngrams)
            for ngram, count in hyp_ngrams_count.items():
                if ngram in ref_ngrams_count:
                    matched += min(count, ref_ngrams_count[ngram])
            
            # 累加分子和分母
            c[n-1] += matched
            t[n-1] += len(hyp_ngrams)
    # 计算各个 n-gram 精确度
    smooth_val = 1.
    for n in range(4):
        if c[n] == 0 :
            smooth_val = smooth_val / 2
            p[n] = 100 * smooth_val / t[n]
        else:
            p[n] = 100 * c[n] / t[n]
    # brevity penalty
    bp = 1.
    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len)
    # 计算 BLEU 分数
    bleu_score = bp * math.exp(sum((1/4) * math.log(pn) for pn in p))

    # ppt bledu score
    r = [c[i]/t[i] for i in range(4)]
    ppt_bleu_score = r[0]*r[1]*r[2]*r[3]/(1-math.log(bp))

    return {
        "ppt_bleu_score": ppt_bleu_score,
        "sacrebleu_score": bleu_score,
        "correct": c,        # list
        "total": t,          # list
        "percent": [100*r[i] for i in range(4)],  # list
        "p_smooth": p,              # list
        "hyps_len": hyp_len, # list
        "ref_len": ref_len,   # list
        "exp_brevity_penalty": bp,
        "linear_brevity_penalty": 1/(1 - math.log(bp))
    }
