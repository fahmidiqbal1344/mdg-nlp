import pandas as pd
import networkx as nx
import os, sys, json, math
import argparse
from itertools import combinations
from collections import defaultdict as ddict
from scipy.stats import spearmanr, kendalltau
import seaborn as sns
import matplotlib.pyplot as plt


# def build_model_score_map(df: pd.DataFrame) -> dict:
#     """
#     Build a model -> score mapping from better_model/better_f1 and worse_model/worse_f1 columns.
#     Assumes each model has a consistent score across rows.
#     """
#     score_map = {}

#     for _, row in df.iterrows():
#         bm, bf1 = row["better_model"], row["better_f1"]
#         wm, wf1 = row["worse_model"], row["worse_f1"]

#         if pd.notna(bm) and pd.notna(bf1):
#             if bm in score_map and abs(score_map[bm] - bf1) > 1e-12:
#                 raise ValueError(
#                     f"Inconsistent score for model {bm}: "
#                     f"{score_map[bm]} vs {bf1}"
#                 )
#             score_map[bm] = float(bf1)

#         if pd.notna(wm) and pd.notna(wf1):
#             if wm in score_map and abs(score_map[wm] - wf1) > 1e-12:
#                 raise ValueError(
#                     f"Inconsistent score for model {wm}: "
#                     f"{score_map[wm]} vs {wf1}"
#                 )
#             score_map[wm] = float(wf1)

#     return score_map

# def build_significance_graph(df: pd.DataFrame, alpha: float = 0.05) -> tuple[nx.DiGraph, dict]:
#     """
#     Build graph where edge A -> B means:
#     A is significantly better than B.
#     """
#     score_map = build_model_score_map(df)

#     # Keep only significant comparisons
#     sig_df = df[df["p_val"] <= alpha].copy()

#     G = nx.DiGraph()
#     G.add_nodes_from(score_map.keys())

#     for _, row in sig_df.iterrows():
#         better = row["better_model"]
#         worse = row["worse_model"]

#         if pd.isna(better) or pd.isna(worse):
#             continue
#         if better == worse:
#             continue

#         G.add_edge(better, worse)

#     return G, score_map


# def tiers_from_dag(G: nx.DiGraph, score_map: dict) -> list[list[str]]:
#     """
#     Compute ordered tiers from a DAG by repeatedly taking nodes with in-degree 0.
#     Models inside a tier are sorted by score descending.
#     """
#     H = G.copy()
#     tiers = []

#     while H.number_of_nodes() > 0:
#         current_tier = [n for n in H.nodes if H.in_degree(n) == 0]

#         if not current_tier:
#             raise ValueError("Graph is not a DAG; cycle detected.")

#         current_tier = sorted(current_tier, key=lambda m: (-score_map[m], m))
#         tiers.append(current_tier)
#         H.remove_nodes_from(current_tier)

#     return tiers


# def tiers_with_scc_collapse(G: nx.DiGraph, score_map: dict) -> list[list[str]]:
#     """
#     Safer version:
#     If cycles exist, collapse strongly connected components (SCCs) first,
#     then compute tiers on the condensed DAG.
#     """
#     if nx.is_directed_acyclic_graph(G):
#         return tiers_from_dag(G, score_map)

#     # Collapse SCCs
#     sccs = list(nx.strongly_connected_components(G))
#     condensed = nx.condensation(G, sccs)

#     # Map component id -> models
#     component_to_models = {}
#     for node_id, members in enumerate(sccs):
#         component_to_models[node_id] = sorted(
#             list(members),
#             key=lambda m: (-score_map[m], m)
#         )

#     H = condensed.copy()
#     tiers = []

#     while H.number_of_nodes() > 0:
#         zero_indegree_components = [n for n in H.nodes if H.in_degree(n) == 0]
#         if not zero_indegree_components:
#             raise ValueError("Condensed graph still has a cycle, which should not happen.")

#         tier_models = []
#         for comp in zero_indegree_components:
#             tier_models.extend(component_to_models[comp])

#         tier_models = sorted(tier_models, key=lambda m: (-score_map[m], m))
#         tiers.append(tier_models)
#         H.remove_nodes_from(zero_indegree_components)

#     return tiers


# def make_tier_dataframe(tiers: list[list[str]], score_map: dict) -> pd.DataFrame:
#     rows = []
#     for tier_id, models in enumerate(tiers, start=1):
#         for rank_in_tier, model in enumerate(models, start=1):
#             rows.append({
#                 "tier": tier_id,
#                 "rank_within_tier": rank_in_tier,
#                 "model": model,
#                 "score": score_map[model],
#             })
#     return pd.DataFrame(rows)

MIN_TIER_SIZE = 1   # set to 1 if you also want singleton tiers


def load_results(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    required_cols = {
        "key1", "key2", "p_val",
        "better_model", "better_f1",
        "worse_model", "worse_f1"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def build_score_map(df: pd.DataFrame, constraints= 'all') -> dict:
    """
    Build model -> performance score from better_model/worse_model columns.
    Checks that each model always has the same score.
    """
    score_map = {}

    for _, row in df.iterrows():
        bm, bf1 = row["better_model"], row["better_f1"]
        wm, wf1 = row["worse_model"], row["worse_f1"]

        if constraints == 'all':
            pass

        elif constraints == 'inductive':
            if pd.notna(bm) and pd.notna(bf1) and 'inductive' not in bm:
                continue
            if pd.notna(wm) and pd.notna(wf1) and 'inductive' not in wm:
                continue
        
        elif constraints == 'non-inductive':
            if pd.notna(bm) and pd.notna(bf1) and 'inductive' in bm:
                continue
            if pd.notna(wm) and pd.notna(wf1) and 'inductive' in wm:
                continue
        

        if pd.notna(bm) and pd.notna(bf1):
            bf1 = float(bf1)
            if bm in score_map and abs(score_map[bm] - bf1) > 1e-12:
                raise ValueError(
                    f"Inconsistent score for model {bm}: "
                    f"{score_map[bm]} vs {bf1}"
                )
            score_map[bm] = bf1

        if pd.notna(wm) and pd.notna(wf1):
            wf1 = float(wf1)
            if wm in score_map and abs(score_map[wm] - wf1) > 1e-12:
                raise ValueError(
                    f"Inconsistent score for model {wm}: "
                    f"{score_map[wm]} vs {wf1}"
                )
            score_map[wm] = wf1

    return score_map


def build_pairwise_pval_map(df: pd.DataFrame) -> dict:
    """
    Build an unordered pair -> p-value map using (key1, key2).
    Since significance-within-tier is undirected, we store pairs canonically.
    """
    pval_map = {}

    for _, row in df.iterrows():
        a = row["key1"]
        b = row["key2"]
        p = float(row["p_val"])

        if pd.isna(a) or pd.isna(b):
            continue
        if a == b:
            continue

        pair = tuple(sorted((a, b)))

        if pair in pval_map:
            # If duplicate rows exist, enforce consistency
            if abs(pval_map[pair] - p) > 1e-12:
                raise ValueError(
                    f"Inconsistent p-value for pair {pair}: "
                    f"{pval_map[pair]} vs {p}"
                )
        else:
            pval_map[pair] = p

    return pval_map


def is_nonsignificant(a: str, b: str, pval_map: dict, alpha: float = 0.05) -> bool:
    """
    True if pair (a,b) is NOT significant.
    """
    if a == b:
        return True

    pair = tuple(sorted((a, b)))
    if pair not in pval_map:
        raise KeyError(f"Missing p-value for pair {pair}")

    return pval_map[pair] > alpha


def tier_is_valid(models_in_tier: list, pval_map: dict, alpha: float = 0.05) -> bool:
    """
    A tier is valid iff every pair inside it is non-significant.
    """
    for a, b in combinations(models_in_tier, 2):
        if not is_nonsignificant(a, b, pval_map, alpha):
            return False
    return True


def build_overlapping_tiers(models_sorted: list, pval_map: dict,
                            alpha: float = 0.05,
                            min_tier_size: int = 2) -> list[list[str]]:
    """
    For each starting position i in the performance-sorted list,
    build the largest consecutive tier [i:j] such that all pairs
    inside the tier are non-significant.

    This produces overlapping tiers and allows repeated models.
    """
    n = len(models_sorted)
    tiers = []

    for start in range(n):
        best_end = start

        for end in range(start + 1, n):
            candidate = models_sorted[start:end + 1]
            if tier_is_valid(candidate, pval_map, alpha):
                best_end = end
            else:
                # Once a consecutive window fails, larger windows will also fail
                break

        tier = models_sorted[start:best_end + 1]

        if len(tier) >= min_tier_size:
            tiers.append(tier)

    return tiers


def remove_duplicate_tiers(tiers: list[list[str]]) -> list[list[str]]:
    """
    Remove exact duplicate tiers while preserving order.
    """
    seen = set()
    unique_tiers = []

    for tier in tiers:
        key = tuple(tier)
        if key not in seen:
            seen.add(key)
            unique_tiers.append(tier)

    return unique_tiers

def remove_completely_overlapping_tiers(tiers: list[list[str]]) -> list[list[str]]:
    """
    Remove tiers if all the models in a given tier is also present in a higher tier. This ensures that each tier has at least one unique model that is not present in any higher tier.

    Args:
        tiers (list[list[str]]): _description_

    Returns:
        list[list[str]]: _description_        
    """
    unique_tiers = []
    seen_models = set()

    for tier in tiers:
        if any(model not in seen_models for model in tier):
            unique_tiers.append(tier)
            seen_models.update(tier)

    return unique_tiers


def make_tier_dataframe(tiers: list[list[str]], score_map: dict) -> pd.DataFrame:
    rows = []
    for tier_id, tier in enumerate(tiers, start=1):
        for rank_in_tier, model in enumerate(tier, start=1):
            rows.append({
                "tier": tier_id,
                "rank_within_tier": rank_in_tier,
                "model": model,
                "score": score_map[model],
            })
    return pd.DataFrame(rows)


def verify_tiers(tiers: list[list[str]], pval_map: dict, alpha: float = 0.05) -> None:
    for idx, tier in enumerate(tiers, start=1):
        if not tier_is_valid(tier, pval_map, alpha):
            raise AssertionError(f"Tier {idx} is invalid: contains a significant pair.")


def produce_tier_df(CSV_FILE, ALPHA=0.05, MIN_TIER_SIZE=1, constraints=None):
    df = load_results(CSV_FILE)
    score_map = build_score_map(df, constraints=constraints)
    pval_map = build_pairwise_pval_map(df)

    # Rank models globally by score
    models_sorted = sorted(score_map.keys(), key=lambda m: (-score_map[m], m))

    print("Global score ranking:")
    for i, m in enumerate(models_sorted, start=1):
        print(f"{i:2d}. {m:40s} {score_map[m]:.4f}")

    # Build overlapping tiers
    tiers = build_overlapping_tiers(
        models_sorted=models_sorted,
        pval_map=pval_map,
        alpha=ALPHA,
        min_tier_size=MIN_TIER_SIZE,
    )

    # tiers = remove_duplicate_tiers(tiers)
    tiers = remove_completely_overlapping_tiers(tiers)
    verify_tiers(tiers, pval_map, ALPHA)

    print("\nOverlapping non-significant tiers:")
    for i, tier in enumerate(tiers, start=1):
        formatted = ", ".join(f"{m} ({score_map[m]:.4f})" for m in tier)
        print(f"Tier {i}: {formatted}")

    
    tier_df = make_tier_dataframe(tiers, score_map)
    # tier_df.to_csv("overlapping_nonsignificant_tiers.csv", index=False)
    # print("\nSaved: overlapping_nonsignificant_tiers.csv")

    return tier_df


def get_ranking():

    for task in ['time', 'event']:

        for constraints in ['all', 'inductive', 'non-inductive']:

            args.task       = task
            args.constraints = constraints

            dimensions       =  ['base', 'length', 'adversarial', 'domain', 'compositional', 'domain_base', 'domain_clinical', 'adversarial_prompt', 'domain_vocab']

            results_dict = ddict(dict)

            for dimension in dimensions:
                csv_file         = f"../analysis_results/sigtest/all_dim_results_{args.task}_{dimension}_sigtest.csv"
                
                tier_df          = produce_tier_df(csv_file, ALPHA=args.alpha, MIN_TIER_SIZE=1, constraints=args.constraints)

                
                tier_df.to_csv(f"../analysis_results/rankings/{args.task}_{dimension}_{args.constraints}_model_tiers.csv", index=False)

                print("\nSaved tier assignments to model_tiers.csv")

                mean_model_tier = tier_df.groupby('model')['tier'].mean().to_dict()

                for model, tier in mean_model_tier.items():
                    results_dict[dimension][model] = tier
            
            
            with open(f'../analysis_results/{args.task}_{args.constraints}_configwise_ranking.json', 'w') as f:
                json.dump(results_dict, f, indent=4)



def get_ranking_correlations():

    for constraints in ['all', 'inductive', 'non-inductive']:

        for task in ['time', 'event']:

            args.task       = task

            args.constraints = constraints

            results_dict = json.load(open(f'../analysis_results/{args.task}_{args.constraints}_configwise_ranking.json', 'r'))

            dimension_dict = {
                'base': 'TB (Base)',
                'length': 'Len',
                'adversarial': 'Adv',
                'domain': 'Dom_Clin',
                'compositional': 'Comp',
                'domain_vocab': 'Dom_Voc',
                'adversarial_prompt': 'Adv_Pmt'
            }
            dimensions = list(dimension_dict.keys())

            corr_dict    = ddict(list)
            sr_dict      = ddict(list)
            kt_dict      = ddict(list)
            pr_dict      = ddict(list)
                    
            for i, dimension1 in enumerate(dimensions):
                for j, dimension2 in enumerate(dimensions):
                    
                    src_accs, tgt_accs = [], []
                    
                    for model in results_dict[dimension1]:
                        src_accs.append(results_dict[dimension1][model])
                        tgt_accs.append(results_dict[dimension2][model])                
                    
                        
                    sr_dict['dimension1'].append(dimension_dict[dimension1])
                    sr_dict['dimension2'].append(dimension_dict[dimension2])
                    sr_dict['corr'].append(spearmanr(src_accs, tgt_accs)[0])
                    sr_dict['pvalue'].append(spearmanr(src_accs, tgt_accs)[1])
                    
                    kt_dict['dimension1'].append(dimension_dict[dimension1])
                    kt_dict['dimension2'].append(dimension_dict[dimension2])
                    kt_dict['corr'].append(kendalltau(src_accs, tgt_accs)[0])
                    kt_dict['pvalue'].append(kendalltau(src_accs, tgt_accs)[1])
                    
                    print(dimension1, dimension2, spearmanr(src_accs, tgt_accs)[0], kendalltau(src_accs, tgt_accs)[0])

            sr_df = pd.DataFrame(sr_dict)
            kt_df = pd.DataFrame(kt_dict)
            
            p_cnts = 0
            for idx, row in sr_df.iterrows():
                if row['pvalue'] < 0.05:
                    p_cnts += 1
            
            print(f'Number of significant correlations: {p_cnts}/ {len(sr_df)}')  
            
            p_cnts = 0
            for idx, row in kt_df.iterrows():
                if row['pvalue'] < 0.05:
                    p_cnts += 1
            
            print(f'Number of significant correlations: {p_cnts}/ {len(kt_df)}')  
            
            sns.heatmap(sr_df.pivot(index='dimension1', columns='dimension2', values='corr'), annot=True, cmap='crest', fmt=".2f")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.savefig(f'../visualizations/{args.task}_{args.constraints}_configwise_sr_rank_reduced.pdf', facecolor='w', bbox_inches="tight")
            plt.clf()
            plt.close()
            
            # how to clear the matplotlib.pyplot 
            
            sns.heatmap(kt_df.pivot(index='dimension1', columns='dimension2', values='corr'), annot=True, cmap='crest', fmt=".2f")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.savefig(f'../visualizations/{args.task}_{args.constraints}_configwise_kt_rank_reduced.pdf', facecolor='w', bbox_inches="tight")
            plt.clf()
            plt.close()




def get_finetuned_ranking():

    for task in ['time', 'event']:

        for constraints in ['all']:

            args.task       = task
            args.constraints = constraints

            dimensions       =  ['base', 'length', 'adversarial', 'domain', 'adversarial_prompt', 'domain_vocab']

            results_dict = ddict(dict)

            for dimension in dimensions:
                try:
                    csv_file         = f"../analysis_results/sigtest/finetuned_all_dim_results_{args.task}_{dimension}_sigtest.csv"
                    
                    tier_df          = produce_tier_df(csv_file, ALPHA=args.alpha, MIN_TIER_SIZE=1, constraints=args.constraints)

                    
                    tier_df.to_csv(f"../analysis_results/rankings/{args.task}_{dimension}_{args.constraints}_finetuned_model_tiers.csv", index=False)

                    # print("\nSaved tier assignments to model_tiers.csv")

                    mean_model_tier = tier_df.groupby('model')['tier'].mean().to_dict()

                    for model, tier in mean_model_tier.items():
                        results_dict[dimension][model] = tier

                except Exception as e:
                    import pdb; pdb.set_trace()

            with open(f'../analysis_results/{args.task}_{args.constraints}_configwise_finetuned_ranking.json', 'w') as f:
                json.dump(results_dict, f, indent=4)



def get_finetuned_ranking_correlations():

    constraints = 'all'

    for task in ['time', 'event']:

        args.task       = task

        args.constraints = constraints

        results_dict = json.load(open(f'../analysis_results/{args.task}_{args.constraints}_configwise_finetuned_ranking.json', 'r'))

        dimension_dict = {
            'base': 'TB (Base)',
            'length': 'Len',
            'adversarial': 'Adv',
            'domain': 'Dom_Clin',
            'domain_vocab': 'Dom_Voc',
            'adversarial_prompt': 'Adv_Pmt'
        }
        dimensions = list(dimension_dict.keys())

        corr_dict    = ddict(list)
        sr_dict      = ddict(list)
        kt_dict      = ddict(list)
        pr_dict      = ddict(list)
                
        for i, dimension1 in enumerate(dimensions):
            for j, dimension2 in enumerate(dimensions):
                
                src_accs, tgt_accs = [], []
                
                for model in results_dict[dimension1]:
                    src_accs.append(results_dict[dimension1][model])
                    tgt_accs.append(results_dict[dimension2][model])                
                
                    
                sr_dict['dimension1'].append(dimension_dict[dimension1])
                sr_dict['dimension2'].append(dimension_dict[dimension2])
                sr_dict['corr'].append(spearmanr(src_accs, tgt_accs)[0])
                sr_dict['pvalue'].append(spearmanr(src_accs, tgt_accs)[1])
                
                kt_dict['dimension1'].append(dimension_dict[dimension1])
                kt_dict['dimension2'].append(dimension_dict[dimension2])
                kt_dict['corr'].append(kendalltau(src_accs, tgt_accs)[0])
                kt_dict['pvalue'].append(kendalltau(src_accs, tgt_accs)[1])
                
                print(dimension1, dimension2, spearmanr(src_accs, tgt_accs)[0], kendalltau(src_accs, tgt_accs)[0])

        sr_df = pd.DataFrame(sr_dict)
        kt_df = pd.DataFrame(kt_dict)
        
        p_cnts = 0
        for idx, row in sr_df.iterrows():
            if row['pvalue'] < 0.05:
                p_cnts += 1
        
        print(f'Number of significant correlations: {p_cnts}/ {len(sr_df)}')  
        
        p_cnts = 0
        for idx, row in kt_df.iterrows():
            if row['pvalue'] < 0.05:
                p_cnts += 1
        
        print(f'Number of significant correlations: {p_cnts}/ {len(kt_df)}')  
        
        sns.heatmap(sr_df.pivot(index='dimension1', columns='dimension2', values='corr'), annot=True, cmap='crest', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(f'../visualizations/{args.task}_{args.constraints}_configwise_finetuned_sr_rank_reduced.pdf', facecolor='w', bbox_inches="tight")
        plt.clf()
        plt.close()
        
        # how to clear the matplotlib.pyplot 
        
        sns.heatmap(kt_df.pivot(index='dimension1', columns='dimension2', values='corr'), annot=True, cmap='crest', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(f'../visualizations/{args.task}_{args.constraints}_configwise_finetuned_kt_rank_reduced.pdf', facecolor='w', bbox_inches="tight")
        plt.clf()
        plt.close()






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Rank models based on pairwise significance tests.") 
    parser.add_argument("--task", type=str, default='time')
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--constraints", type=str, default='all', help="Optional constraints for score map construction (e.g., 'inductive').")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for determining tiers.")

    args = parser.parse_args()    

    if args.step == 'rank':
        get_ranking()
    
    if args.step == 'rank_corr':
        get_ranking_correlations()

    if args.step == 'ft_rank':
        get_finetuned_ranking()
    
    if args.step == 'ft_rank_corr':
        get_finetuned_ranking_correlations()

