from __future__ import annotations
from typing import List, Dict, Tuple
from collections import Counter

def _normalize_label(label: str) -> str:
    if label is None:
        return ""
    # normalize like your evaluator expects
    return str(label).strip().upper().replace(" ", "_").replace("-", "_")

def evaluate_classification(gold: List[str], pred: List[str]) -> Dict[str, float | Dict]:
    """
    Single-label multiclass classification metrics:
      - accuracy
      - macro_precision, macro_recall, macro_f1   (EXCLUDES classes with support == 0)
      - micro_precision, micro_recall, micro_f1   (computed over classes present in gold)
      - weighted_f1                               (weighted by gold support)
      - per_class breakdown (precision/recall/f1/support)
    """
    gold = [_normalize_label(g) for g in gold]
    pred = [_normalize_label(p) for p in pred]
    assert len(gold) == len(pred), "gold and pred must have same length"

    # Accuracy
    correct = sum(g == p for g, p in zip(gold, pred))
    accuracy = correct / len(gold) if gold else 0.0

    # Per-class counts
    labels = sorted(set(gold) | set(pred))
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        support[g] += 1
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    # Only classes that actually appear in GOLD (support > 0)
    labels_in_gold = [c for c in labels if support[c] > 0]

    # Per-class metrics
    class_precisions, class_recalls, class_f1s = [], [], []
    per_class: Dict[str, Dict[str, float]] = {}
    for c in labels:
        # report per-class even if support==0, but it won't affect macro (excluded)
        p_c = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r_c = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) else 0.0
        per_class[c] = {"support": support[c], "precision": p_c, "recall": r_c, "f1": f1_c}

        if support[c] > 0:  # EXCLUDE zero-support classes from macro
            class_precisions.append(p_c)
            class_recalls.append(r_c)
            class_f1s.append(f1_c)

    macro_precision = (sum(class_precisions) / len(class_precisions)) if class_precisions else 0.0
    macro_recall    = (sum(class_recalls)    / len(class_recalls))    if class_recalls    else 0.0
    macro_f1        = (sum(class_f1s)       / len(class_f1s))         if class_f1s       else 0.0

    # Micro metrics over classes present in gold (matches sklearn's default behavior)
    TP = sum(tp[c] for c in labels_in_gold)
    FP = sum(fp[c] for c in labels_in_gold)
    FN = sum(fn[c] for c in labels_in_gold)

    micro_precision = TP / (TP + FP) if (TP + FP) else 0.0
    micro_recall    = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) else 0.0)

    # Weighted-F1 (weights by gold support)
    total_support = sum(support[c] for c in labels_in_gold)
    if total_support > 0:
        weighted_f1 = sum(per_class[c]["f1"] * support[c] for c in labels_in_gold) / total_support
    else:
        weighted_f1 = 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }