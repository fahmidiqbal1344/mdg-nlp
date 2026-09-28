from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------- normalization & tokenization ----------
def _preprocess(text: str) -> str:
    
    return text.lower() if text else ""

def _tokens(s: str) -> List[str]:
    
    s = _preprocess(s)
    return s.split()

def _safe_max(vals: Iterable[float]) -> float:
    """max() with 0.0 default for empty iterables."""
    return max(vals, default=0.0)


def _lcs_len_char(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        new_dp = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                new_dp[j] = dp[j - 1] + 1
                if new_dp[j] > best:
                    best = new_dp[j]
        dp = new_dp
    return best

def _lcs_len_word(a: str, b: str) -> int:
    ta, tb = _tokens(a), _tokens(b)
    la, lb = len(ta), len(tb)
    dp = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        new_dp = [0] * (lb + 1)
        ai = ta[i - 1]
        for j in range(1, lb + 1):
            if ai == tb[j - 1]:
                new_dp[j] = dp[j - 1] + 1
                if new_dp[j] > best:
                    best = new_dp[j]
        dp = new_dp
    return best

def _num_chars(x: str) -> int:
    
    return len(x) if isinstance(x, str) else len(str(x))

def _num_words(x: str) -> int:
    
    x = x if isinstance(x, str) else str(x)
    return len(x.split())

def _dedup_preserve_order(spans: List[str]) -> List[str]:
    
    return list(spans or [])


def _partial_micro_metrics(
    data: List[Tuple[List[str], List[str]]],
    lcs_func: Callable[[str, str], int],
    length_func: Callable[[str], int],
    preprocess: Callable[[str], str],
) -> Tuple[float, float, float]:
    s_prec: List[float] = []
    s_rec: List[float] = []

    for gold_spans_raw, pred_spans_raw in data:
        gold_spans = _dedup_preserve_order(gold_spans_raw)
        pred_spans = _dedup_preserve_order(pred_spans_raw)

        # precision contributions
        for p in pred_spans:
            pp = preprocess(p)
            if not pp:
                s_prec.append(0.0); continue
            denom = length_func(p)
            if denom <= 0 or not gold_spans:
                s_prec.append(0.0); continue
            best = _safe_max((lcs_func(pp, preprocess(g)) / denom) for g in gold_spans)
            s_prec.append(best)

        # recall contributions
        for g in gold_spans:
            gg = preprocess(g)
            if not gg:
                s_rec.append(0.0); continue
            denom = length_func(g)
            if denom <= 0 or not pred_spans:
                s_rec.append(0.0); continue
            best = _safe_max((lcs_func(gg, preprocess(p)) / denom) for p in pred_spans)
            s_rec.append(best)

    precision = sum(s_prec) / len(s_prec) if s_prec else 0.0
    recall    = sum(s_rec) / len(s_rec) if s_rec else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

# ---------- Hungarian (one-to-one) ----------
def _matching_micro_metrics(
    data: List[Tuple[List[str], List[str]]],
    lcs_func: Callable[[str, str], int],
    pred_len_func: Callable[[str], int],
    gold_len_func: Callable[[str], int],
    preprocess: Callable[[str], str],
) -> Tuple[float, float, float]:
    total_preds = 0
    total_golds = 0
    sum_prec = 0.0
    sum_rec  = 0.0

    for gold_spans_raw, pred_spans_raw in data:
        gold_spans = _dedup_preserve_order(gold_spans_raw)
        pred_spans = _dedup_preserve_order(pred_spans_raw)

        nP, nG = len(pred_spans), len(gold_spans)
        total_preds += nP
        total_golds += nG
        if nP == 0 or nG == 0:
            continue

        P_texts = [preprocess(p) for p in pred_spans]
        G_texts = [preprocess(g) for g in gold_spans]

        P = np.zeros((nP, nG), dtype=float)
        for i, pp in enumerate(P_texts):
            denom_p = pred_len_func(pp) if pp else 0
            if denom_p <= 0: continue
            for j, gg in enumerate(G_texts):
                if not gg: continue
                num = lcs_func(pp, gg)
                if num > 0:
                    P[i, j] = num / denom_p

        R = np.zeros((nP, nG), dtype=float)
        for i, pp in enumerate(P_texts):
            for j, gg in enumerate(G_texts):
                denom_g = gold_len_func(gg) if gg else 0
                if denom_g <= 0: continue
                num = lcs_func(pp, gg)
                if num > 0:
                    R[i, j] = num / denom_g

        if P.size:
            r, c = linear_sum_assignment(-P)
            sum_prec += float(P[r, c].sum())
        if R.size:
            r, c = linear_sum_assignment(-R)
            sum_rec += float(R[r, c].sum())

    precision = (sum_prec / total_preds) if total_preds > 0 else 0.0
    recall    = (sum_rec  / total_golds) if total_golds > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------- Hungarian (one-to-one) ----------
def _instancewise_matching_micro_metrics(
    data: List[Tuple[List[str], List[str]]],
    lcs_func: Callable[[str, str], int],
    pred_len_func: Callable[[str], int],
    gold_len_func: Callable[[str], int],
    preprocess: Callable[[str], str],
) -> Tuple[float, float, float]:
    total_preds = 0
    total_golds = 0
    sum_prec = 0.0
    sum_rec  = 0.0

    instance_wise_scores = []

    for gold_spans_raw, pred_spans_raw in data:
        gold_spans = _dedup_preserve_order(gold_spans_raw)
        pred_spans = _dedup_preserve_order(pred_spans_raw)

        nP, nG = len(pred_spans), len(gold_spans)
        total_preds += nP
        total_golds += nG

        if nP == 0 or nG == 0:
            instance_wise_scores.append({'sum_prec': 0.0, 'sum_rec': 0.0, 'nP': nP, 'nG': nG})
            continue

        P_texts = [preprocess(p) for p in pred_spans]
        G_texts = [preprocess(g) for g in gold_spans]

        P = np.zeros((nP, nG), dtype=float)
        for i, pp in enumerate(P_texts):
            denom_p = pred_len_func(pp) if pp else 0
            if denom_p <= 0: continue
            for j, gg in enumerate(G_texts):
                if not gg: continue
                num = lcs_func(pp, gg)
                if num > 0:
                    P[i, j] = num / denom_p

        R = np.zeros((nP, nG), dtype=float)
        for i, pp in enumerate(P_texts):
            for j, gg in enumerate(G_texts):
                denom_g = gold_len_func(gg) if gg else 0
                if denom_g <= 0: continue
                num = lcs_func(pp, gg)
                if num > 0:
                    R[i, j] = num / denom_g

        curr_prec, curr_rec = 0.0, 0.0
        if P.size:
            r, c = linear_sum_assignment(-P)
            sum_prec += float(P[r, c].sum())
            curr_prec = float(P[r, c].sum())

        if R.size:
            r, c = linear_sum_assignment(-R)
            sum_rec += float(R[r, c].sum())
            curr_rec = float(R[r, c].sum())
        
        instance_wise_scores.append({'sum_prec': curr_prec , 'sum_rec': curr_rec, 'nP': nP, 'nG': nG})

    # precision = (sum_prec / total_preds) if total_preds > 0 else 0.0
    # recall    = (sum_rec  / total_golds) if total_golds > 0 else 0.0
    # f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    # return precision, recall, f1

    return instance_wise_scores



def instancewise_metrics(
        gold: List[List[str]],
        pred: List[List[str]],
        *,
        case_insensitive: bool = True,
        strip: bool = True,
    ) -> Dict[str, float]:

    """
    Returns all 12 metrics: Best-of (partial) + Hungarian, at char/word levels.
    """
    def preprocess(s: str) -> str:
        if s is None:
            return ""
        
        return s.lower() if case_insensitive else s

    data = list(zip(gold, pred))

    instancewise_match_char = _instancewise_matching_micro_metrics(data, _lcs_len_char, _num_chars, _num_chars, preprocess)
    instancewise_match_word = _instancewise_matching_micro_metrics(data, _lcs_len_word, _num_words, _num_words, preprocess)

    return {
        "instancewise_match_char": instancewise_match_char,
        "instancewise_match_word": instancewise_match_word,
    }


def evaluate_partial_and_match(
    gold: List[List[str]],
    pred: List[List[str]],
    *,
    case_insensitive: bool = True,
    strip: bool = True,
) -> Dict[str, float]:
    """
    Returns all 12 metrics: Best-of (partial) + Hungarian, at char/word levels.
    """
    def preprocess(s: str) -> str:
        if s is None:
            return ""
        
        return s.lower() if case_insensitive else s

    data = list(zip(gold, pred))

    # Best-of partial
    cP, cR, cF = _partial_micro_metrics(data, _lcs_len_char, _num_chars, preprocess)
    wP, wR, wF = _partial_micro_metrics(data, _lcs_len_word, _num_words, preprocess)

    # Hungarian 1:1
    mcP, mcR, mcF = _matching_micro_metrics(data, _lcs_len_char, _num_chars, _num_chars, preprocess)
    mwP, mwR, mwF = _matching_micro_metrics(data, _lcs_len_word, _num_words, _num_words, preprocess)

    return {
        "char_precision": cP,  "char_recall": cR,  "char_f1": cF,
        "word_precision": wP,  "word_recall": wR,  "word_f1": wF,
        "char_precision_match": mcP, "char_recall_match": mcR, "char_f1_match": mcF,
        "word_precision_match": mwP, "word_recall_match": mwR, "word_f1_match": mwF,
    }


def agg_scores(df, model1, model2):

    # mistral7b_abductive_nG,mistral7b_deductive,mistral7b_deductive_prec,mistral7b_deductive_rec,mistral7b_deductive_nP

    model1_prec, model1_rec, model1_f1, model1_nP, model1_nG = 0.0, 0.0, 0.0, 0, 0
    model2_prec, model2_rec, model2_f1, model2_nP, model2_nG = 0.0, 0.0, 0.0, 0, 0

    
    name1, rtype1 = model1.split('_')
    name2, rtype2 = model2.split('_')

    for idx, row in df.iterrows():

        model1_prec += row[f'{name1}_{rtype1}_prec']
        model1_rec  += row[f'{name1}_{rtype1}_rec']
        model1_nP   += row[f'{name1}_{rtype1}_nP']
        model1_nG   += row[f'{name1}_{rtype1}_nG']

        model2_prec += row[f'{name2}_{rtype2}_prec']
        model2_rec  += row[f'{name2}_{rtype2}_rec']
        model2_nP   += row[f'{name2}_{rtype2}_nP']
        model2_nG   += row[f'{name2}_{rtype2}_nG']
    
    sum_model1_prec = (model1_prec / model1_nP) if model1_nP > 0 else 0.0
    sum_model1_rec  = (model1_rec  / model1_nG) if model1_nG > 0 else 0.0
    sum_model1_f1   = (2 * sum_model1_prec * sum_model1_rec) / (sum_model1_prec + sum_model1_rec) if (sum_model1_prec + sum_model1_rec) else 0.0

    sum_model2_prec = (model2_prec / model2_nP) if model2_nP > 0 else 0.0
    sum_model2_rec  = (model2_rec  / model2_nG) if model2_nG > 0 else 0.0
    sum_model2_f1   = (2 * sum_model2_prec * sum_model2_rec) / (sum_model2_prec + sum_model2_rec) if (sum_model2_prec + sum_model2_rec) else 0.0


    return sum_model1_f1, sum_model2_f1


def agg_ft_scores(df, model1, model2):

    # mistral7b_abductive_nG,mistral7b_deductive,mistral7b_deductive_prec,mistral7b_deductive_rec,mistral7b_deductive_nP

    model1_prec, model1_rec, model1_f1, model1_nP, model1_nG = 0.0, 0.0, 0.0, 0, 0
    model2_prec, model2_rec, model2_f1, model2_nP, model2_nG = 0.0, 0.0, 0.0, 0, 0

    for idx, row in df.iterrows():

        model1_prec += row[f'{model1}_prec']
        model1_rec  += row[f'{model1}_rec']
        model1_nP   += row[f'{model1}_nP']
        model1_nG   += row[f'{model1}_nG']

        model2_prec += row[f'{model2}_prec']
        model2_rec  += row[f'{model2}_rec']
        model2_nP   += row[f'{model2}_nP']
        model2_nG   += row[f'{model2}_nG']
    
    sum_model1_prec = (model1_prec / model1_nP) if model1_nP > 0 else 0.0
    sum_model1_rec  = (model1_rec  / model1_nG) if model1_nG > 0 else 0.0
    sum_model1_f1   = (2 * sum_model1_prec * sum_model1_rec) / (sum_model1_prec + sum_model1_rec) if (sum_model1_prec + sum_model1_rec) else 0.0

    sum_model2_prec = (model2_prec / model2_nP) if model2_nP > 0 else 0.0
    sum_model2_rec  = (model2_rec  / model2_nG) if model2_nG > 0 else 0.0
    sum_model2_f1   = (2 * sum_model2_prec * sum_model2_rec) / (sum_model2_prec + sum_model2_rec) if (sum_model2_prec + sum_model2_rec) else 0.0


    return sum_model1_f1, sum_model2_f1

    
