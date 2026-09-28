import os
import json
import re
from glob import glob
from itertools import combinations
from collections import defaultdict as ddict
import pandas as pd
import multiprocessing as mp
from eval_script import *
import random


# ---------------------------------------------------------------------
# Worker globals (each process loads df once)
# ---------------------------------------------------------------------
_GLOBAL_DF = None

def _init_sigtest_worker(df_path: str):
    """Initializer: load the big CSV once per worker process."""
    global _GLOBAL_DF
    _GLOBAL_DF = pd.read_csv(df_path)

def _safe_slug(s: str) -> str:
    """Filesystem-safe, reasonably compact identifier."""
    s = str(s)
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    return s[:200]

def _case_to_filename(task: str, dimension: str, key1, key2) -> str:
    # keys are tuples like ('llama17b', 'abductive') in your code :contentReference[oaicite:2]{index=2}
    k1 = _safe_slug(f"{key1[0]}__{key1[1]}")
    k2 = _safe_slug(f"{key2[0]}__{key2[1]}")
    return f"sigtest__task={_safe_slug(task)}__dim={_safe_slug(dimension)}__k1={k1}__k2={k2}.json"

def _atomic_write_json(obj: dict, path: str):
    """Atomic-ish write: write temp then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _run_one_sigtest(args):
    """
    One independent job:
      - filter df to (task, dimension)
      - run compute_bootstrapped_df(curr_df, key1, key2, num_iter)
      - write to a unique per-comparison json file
    """
    global _GLOBAL_DF
    (task, dimension, key1, key2, out_dir, num_iter) = args

    out_path = os.path.join(out_dir, _case_to_filename(task, dimension, key1, key2))
    if os.path.exists(out_path):
        return out_path  # already done

    curr_df = _GLOBAL_DF[(_GLOBAL_DF["task"] == task) & (_GLOBAL_DF["dimension"] == dimension)]

    # Your original compare_results() calls compute_bootstrapped_df(curr_df, key1, key2) :contentReference[oaicite:3]{index=3}
    sigtest_results = compute_bootstrapped_df(curr_df, key1, key2, num_iter=num_iter)

    payload = {
        "task": task,
        "dimension": dimension,
        "key1": list(key1),  # json-friendly
        "key2": list(key2),
        **sigtest_results,
    }
    _atomic_write_json(payload, out_path)
    return out_path

def merge_sigtest_jsons_to_csv(sigtest_dir: str, out_csv_path: str):
    """Utility: combine per-comparison JSON files into one CSV."""
    rows = []
    for fp in sorted(glob(os.path.join(sigtest_dir, "sigtest__*.json"))):
        with open(fp, "r") as f:
            rows.append(json.load(f))
    if not rows:
        print(f"No sigtest json files found in: {sigtest_dir}")
        return
    df = pd.DataFrame(rows)

    # restore tuple-like display for keys if you want
    # (optional) df["key1"] = df["key1"].apply(tuple); df["key2"] = df["key2"].apply(tuple)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f"Wrote merged CSV: {out_csv_path} ({len(df)} rows)")



def compute_bootstrapped_df(df, model1, model2, num_iter = 100):
    
        
    gold_results  = df['gold'].tolist()
    pred_results1 = df[f'{model1}'].tolist()
    pred_results2 = df[f'{model2}'].tolist()

    model1_f1 = evaluate_partial_and_match(gold_results, pred_results1)['word_f1_match']
    model2_f1 = evaluate_partial_and_match(gold_results, pred_results2)['word_f1_match']

    idx_list = [idx for idx in range(len(df))]
            
    sys_diff      = model1_f1 - model2_f1
    
    if sys_diff    <= 0:        
        temp_model = model1
        model1     = model2
        model2     = temp_model
        temp_f1    = model1_f1
        model1_f1  = model2_f1
        model2_f1  = temp_f1
        sys_diff   = -sys_diff    
    
    assert sys_diff > 0

    cnt = 0
    
    for i in range(0, num_iter):
        random.seed(i)
        np.random.seed(i)
        curr_idx_list = random.choices(idx_list, k=100)
        
        pred_model1 = df.iloc[curr_idx_list][f'{model1}'].tolist()
        pred_model2 = df.iloc[curr_idx_list][f'{model2}'].tolist()
        
        curr_golds       = df.iloc[curr_idx_list]['gold'].tolist()
        
        curr_model1_f1   = evaluate_partial_and_match(curr_golds, pred_model1)['word_f1_match']
        curr_model2_f1   = evaluate_partial_and_match(curr_golds, pred_model2)['word_f1_match']
        
        # comes from the 2012 non-paramteric method of boostrap significance testing
        curr_diff        = curr_model1_f1 - curr_model2_f1
        if curr_diff > 2*sys_diff:
            cnt += 1
        
        print(f'Done for {i} for {model1} and {model2}')
            
    
    sig_val = ''
    p_val = cnt/num_iter

    if p_val < 0.001 :
        sig_val = '***'
    elif p_val < 0.01: 
        sig_val = '**'
    elif p_val < 0.05:
        sig_val = '*'
    
    # if p_val > 0.999:
    #     sig_val = '---'
    # elif p_val > 0.99:
    #     sig_val = '--'
    # elif p_val > 0.95:
    #     sig_val = '-'
    
    results_dict = {
        'p_val': p_val, 
        'sig_val': sig_val,
        'better_model': model1,
        'better_f1': round(model1_f1,4),
        'worse_model': model2,
        'worse_f1': round(model2_f1,4)
    }
    
    return results_dict





def compare_results(
    df_path: str = "../results/all_dim_results.csv",
    out_dir: str = "../results/sigtests_by_pair",
    num_iter: int = 100,
    workers: int | None = None,
    chunksize: int = 10,
):
    """
    Parallel version of compare_results():
      - each (task, dimension, key1, key2) is a separate job
      - results written to individual JSON files (safe for parallel writes)
      - resume supported: existing json files are skipped
    """
    # These lists match your original compare_results() :contentReference[oaicite:4]{index=4}
    models = ["llama8b", "llama17b", "mistral7b", "mistral13b", "qwen7b", "qwen22b"]
    reasoning_types = [
        "abductive", "deductive", "inductive",
        "deductive+abductive", "inductive+abductive",
        "inductive+deductive", "inductive+deductive+abductive"
    ]
    time_datas = ["event", "time"]
    dimensions = ["base", "length", "adversarial", "domain"]

    # Build keys as in your original code (skip llama8b+inductive) :contentReference[oaicite:5]{index=5}
    keys = []
    for model in models:
        for r in reasoning_types:
            if model == "llama8b" and r == "inductive":
                continue
            keys.append((model, r))

    os.makedirs(out_dir, exist_ok=True)

    # Enumerate all jobs
    jobs = []
    for task in time_datas:
        for dim in dimensions:
            for key1, key2 in combinations(keys, 2):
                out_path = os.path.join(out_dir, _case_to_filename(task, dim, key1, key2))
                if os.path.exists(out_path):
                    continue
                jobs.append((task, dim, key1, key2, out_dir, num_iter))

    if not jobs:
        print(f"Nothing to do. All comparisons already exist under: {out_dir}")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 2)//2 - 1)

    print(f"Queued {len(jobs)} comparisons. workers={workers}, num_iter={num_iter}")
    print(f"Writing one file per comparison under: {out_dir}")

    # Use "spawn" for portability (macOS), but this still loads df only once per worker via initializer.
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_sigtest_worker,
        initargs=(df_path,),
    ) as pool:
        done = 0
        for _ in pool.imap_unordered(_run_one_sigtest, jobs, chunksize=chunksize):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"Completed {done}/{len(jobs)}", end="\r")
    print("\nAll jobs completed.")

    # Optional: merge everything into a single CSV at the end
    merge_sigtest_jsons_to_csv(out_dir, "../results/all_dim_results_sigtest_merged.csv")



if __name__ == "__main__":
    compare_results()