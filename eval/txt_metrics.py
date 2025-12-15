import numpy as np
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import math


def ensure_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet  # noqa: F401
    except LookupError:
        nltk.download('wordnet')



def get_ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))
    

def compute_tf_idf(cand_ngrams, ref_ngrams, doc_freq, n_refs):
    vocab = set(cand_ngrams.keys()) | set(ref_ngrams.keys())
    vec1, vec2 = [], []
    
    for gram in vocab:
        tf1 = cand_ngrams[gram]
        tf2 = ref_ngrams[gram]
        idf = math.log(n_refs / (doc_freq[gram] + 1))
        vec1.append(tf1 * idf)
        vec2.append(tf2 * idf)
        
    return vec1, vec2


def cosine_sim(vec1, vec2):
    norm1 = math.sqrt(sum([x*x for x in vec1]))
    norm2 = math.sqrt(sum([x*x for x in vec2]))
    dot_product = sum([x*y for x,y in zip(vec1, vec2)])
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0


def compute_bleu_n(candidate, reference, n=4):
    bleu_smooth = SmoothingFunction().method1
    weights = tuple([1.0/n] * n)
    cand_tokens = candidate.split()
    ref_tokens = [reference.split()]
    return sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=bleu_smooth)


def compute_meteor(candidate, reference):
    cand_tokens = candidate.split()
    ref_tokens = [reference.split()]
    return meteor_score(ref_tokens, cand_tokens)


def compute_rougel(candidate, reference):
    rougel_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return rougel_scorer.score(reference, candidate)['rougeL'].fmeasure


def compute_cider(candidates, references):
    n_grams = [1, 2, 3, 4]
    weights = [1/4] * 4
    scores = []

    if not isinstance(candidates, (list, tuple)):
        candidates = [candidates]
    if not isinstance(references, (list, tuple)):
        references = [references]
    
    for n in n_grams:
        doc_freq = Counter()
        for ref in references:
            tokens = ref.split()
            doc_freq.update(get_ngrams(tokens, n).keys())
            
        cider_scores = []
        for cand, ref in zip(candidates, references):
            cand_tokens = cand.split()
            ref_tokens = ref.split()
            
            cand_ngrams = get_ngrams(cand_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            vec1, vec2 = compute_tf_idf(cand_ngrams, ref_ngrams, doc_freq, len(references))
            score = cosine_sim(vec1, vec2)
            cider_scores.append(score)
            
        scores.append(np.mean(cider_scores))
    
    return sum([s * w for s, w in zip(scores, weights)])