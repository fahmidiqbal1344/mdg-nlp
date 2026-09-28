"""
nli models
"""
import os.path
import re
import evaluate
import numpy as np
from collections import defaultdict as ddict
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import  accuracy_score, f1_score
import random
import math
from pprint import pprint
import pickle
from scipy.stats import spearmanr, kendalltau, pearsonr
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import json
from evaluate import load
from mc4.algorithm import mc4_aggregator



# def compute_sig_change_stances():
    
#     sig_df = pd.read_csv(f'../analysis_results/{args.task}_sig_test.csv')
    
#     init_better         = {}
#     init_same           = {}
    
#     for index, row in sig_df.iterrows():
#         dataset             = row['dataset']
#         if dataset != 'squad': continue
#         p_val               = row['p_val']
#         better_model        = row['better_model']
#         worse_model         = row['worse_model']
        
#         if p_val <= 0.05:
#             init_better[(better_model, worse_model)] = {'better':0, 'worse':0, 'equal':0}
#         else:
#             init_same[(better_model, worse_model)] = {'better':0, 'worse':0, 'equal':0}


#     for index, row in sig_df.iterrows():
#         dataset             = row['dataset']
        
#         if dataset == 'squad': continue
        
#         p_val               = row['p_val']
#         better_model        = row['better_model']
#         worse_model         = row['worse_model']
        
#         if p_val <= 0.05:
            
#             if (better_model, worse_model) in init_better:
#                 init_better[(better_model, worse_model)]['better'] += 1
#             elif (worse_model, better_model) in init_better:
#                 init_better[(worse_model, better_model)]['worse'] += 1
#                 print(f'{better_model} is now better than {worse_model} for {dataset}')
            
#             elif (better_model, worse_model) in init_same:
#                 init_same[(better_model, worse_model)]['better'] += 1
#             elif (worse_model, better_model) in init_same:
#                 init_same[(worse_model, better_model)]['worse'] += 1
                
            
#         else:
#             if (better_model, worse_model) in init_better:
#                 init_better[(better_model, worse_model)]['equal'] += 1
#             elif (worse_model, better_model) in init_better:
#                 init_better[(worse_model, better_model)]['equal'] += 1
            
#             elif (better_model, worse_model) in init_same:
#                 init_same[(better_model, worse_model)]['equal'] += 1
#             elif (worse_model, better_model) in init_same:
#                 init_same[(worse_model, better_model)]['equal'] += 1

#     init_better_cnt = ddict(int)
#     init_same_cnt   = ddict(int)
    
#     for model_pair in init_better:
#         init_better_cnt['better'] += init_better[model_pair]['better']
#         init_better_cnt['worse'] += init_better[model_pair]['worse']
#         init_better_cnt['equal'] += init_better[model_pair]['equal']
    
#     for model_pair in init_same:
#         init_same_cnt['better'] += init_same[model_pair]['better']
#         init_same_cnt['worse'] += init_same[model_pair]['worse']
#         init_same_cnt['equal'] += init_same[model_pair]['equal']    

#     pprint(init_better_cnt)
#     pprint(init_same_cnt)
    

def get_model_name(model_rt_name):

    model, reasoning = model_rt_name.split('_')[0], model_rt_name.split('_')[1]

    reverse_reasoning_dict ={
        'abductive': 'A',
        'deductive': 'D',
        'inductive': 'I',
        'deductive+abductive': 'D+A',
        'inductive+abductive': 'I+A',
        'inductive+deductive': 'I+D',
        'inductive+deductive+abductive': 'I+D+A'
    }

    return f'{model}_{reverse_reasoning_dict[reasoning]}'



def validate_hyps():

    models            = ['llama8b', 'llama17b', 'mistral8b', 'mistral13b', 'qwen7b', 'qwen22b']

    reasoning_types   = ['abductive', 'deductive', 'inductive', 'deductive+abductive', 'inductive+abductive', 'inductive+deductive' , 'inductive+deductive+abductive']

    dimensions       =  ['base', 'length', 'adversarial', 'adversarial_prompt', 'domain', 'compositional', 'domain_base', 'domain_clinical', 'domain_vocab']
    tasks            =  ['time', 'event']


    hypothesis_cnt_dict = ddict(lambda: ddict(int))

    model_rts      = ['llama8b_A', 'llama8b_D', 'llama8b_I', 'llama8b_D+A', 'llama8b_I+A', 'llama8b_I+D', 'llama8b_I+D+A', 'llama17b_A', 'llama17b_D', 'llama17b_I', 'llama17b_D+A', 'llama17b_I+A', 'llama17b_I+D', 'llama17b_I+D+A', 'mistral8b_A', 'mistral8b_D', 'mistral8b_I', 'mistral8b_D+A', 'mistral8b_I+A', 'mistral8b_I+D', 'mistral8b_I+D+A', 'mistral13b_A', 'mistral13b_D', 'mistral13b_I', 'mistral13b_D+A', 'mistral13b_I+A', 'mistral13b_I+D', 'mistral13b_I+D+A', 'qwen7b_A', 'qwen7b_D', 'qwen7b_I', 'qwen7b_D+A', 'qwen7b_I+A', 'qwen7b_I+D', 'qwen7b_I+D+A', 'qwen22b_A', 'qwen22b_D', 'qwen22b_I', 'qwen22b_D+A', 'qwen22b_I+A', 'qwen22b_I+D', 'qwen22b_I+D+A']


    dense_models   = ['llama8b_A', 'llama8b_D', 'llama8b_I', 'llama8b_D+A', 'llama8b_I+A', 'llama8b_I+D', 'llama8b_I+D+A',\
        'mistral8b_A', 'mistral8b_D', 'mistral8b_I', 'mistral8b_D+A', 'mistral8b_I+A', 'mistral8b_I+D', 'mistral8b_I+D+A',\
        'qwen7b_A', 'qwen7b_D', 'qwen7b_I', 'qwen7b_D+A', 'qwen7b_I+A', 'qwen7b_I+D', 'qwen7b_I+D+A']

    moe_models   = ['llama17b_A', 'llama17b_D', 'llama17b_I', 'llama17b_D+A', 'llama17b_I+A', 'llama17b_I+D', 'llama17b_I+D+A',\
         'mistral13b_A', 'mistral13b_D', 'mistral13b_I', 'mistral13b_D+A', 'mistral13b_I+A', 'mistral13b_I+D', 'mistral13b_I+D+A',\
         'qwen22b_A', 'qwen22b_D', 'qwen22b_I', 'qwen22b_D+A', 'qwen22b_I+A', 'qwen22b_I+D', 'qwen22b_I+D+A']



    D_models       = ['llama17b_D', 'mistral13b_D', 'qwen22b_D', 'llama8b_D', 'mistral8b_D', 'qwen7b_D']
    I_models       = ['llama17b_I', 'mistral13b_I', 'qwen22b_I', 'llama8b_I', 'mistral8b_I', 'qwen7b_I']
    A_models       = ['llama17b_A', 'mistral13b_A', 'qwen22b_A', 'llama8b_A', 'mistral8b_A', 'qwen7b_A']
    I_D_models     = ['llama17b_I+D', 'mistral13b_I+D', 'qwen22b_I+D', 'llama8b_I+D', 'mistral8b_I+D', 'qwen7b_I+D']
    D_A_models     = ['llama17b_D+A', 'mistral13b_D+A', 'qwen22b_D+A', 'llama8b_D+A', 'mistral8b_D+A', 'qwen7b_D+A']
    I_A_models     = ['llama17b_I+A', 'mistral13b_I+A', 'qwen22b_I+A', 'llama8b_I+A', 'mistral8b_I+A', 'qwen7b_I+A']
    I_D_A_models   = ['llama17b_I+D+A', 'mistral13b_I+D+A', 'qwen22b_I+D+A', 'llama8b_I+D+A', 'mistral8b_I+D+A', 'qwen7b_I+D+A']

    reasoning_dict = {'D': D_models, 'I': I_models, 'A': A_models, 'I+D': I_D_models, 'D+A': D_A_models, 'I+A': I_A_models, 'I+D+A': I_D_A_models}

    reasonings                          = list(reasoning_dict.keys())

    hypothesis_cnt_dict                 = ddict(lambda: ddict(lambda: ddict(lambda: ddict(int))))

    for task in tasks:
        for dimension in dimensions:

            results_dict                = {}
            sigtest_df                  = pd.read_csv(f'../analysis_results/sigtest/all_dim_results_{task}_{dimension}_sigtest.csv')

            for idx, row in sigtest_df.iterrows():
                key1, key2  = row['key1'], row['key2']
                m1, m2      = get_model_name(key1), get_model_name(key2)

                if row['better_model'] == key1:
                    results_dict[(m1,m2)] = (m1, m2, 1 if row['p_val'] <= 0.05 else 0)
                    results_dict[(m2,m1)] = (m1, m2, 1 if row['p_val'] <= 0.05 else 0)

                elif row['better_model'] == key2:
                    results_dict[(m1,m2)] = (m2, m1, 1 if row['p_val'] <= 0.05 else 0)
                    results_dict[(m2,m1)] = (m2, m1, 1 if row['p_val'] <= 0.05 else 0)

            print(len(results_dict))

            # for now check if it is simply larger or smaller or equal.

            for dm, mm in zip(dense_models, moe_models):

                sigval = results_dict[(dm, mm)][2]

                if sigval == 0:
                    hypothesis_cnt_dict['Dense vs MOE'][task][dimension]['equal'] += 1
                    hypothesis_cnt_dict['Dense vs MOE'][task]['All']['equal'] += 1
                
                else:
                    if results_dict[(dm, mm)][0] == dm:
                        hypothesis_cnt_dict['Dense vs MOE'][task][dimension]['better'] += 1
                        hypothesis_cnt_dict['Dense vs MOE'][task]['All']['better'] += 1

                    elif results_dict[(dm, mm)][0] == mm:

                        hypothesis_cnt_dict['Dense vs MOE'][task][dimension]['worse'] += 1
                        hypothesis_cnt_dict['Dense vs MOE'][task]['All']['worse'] += 1
                
                    
                
            for idx1, reasoning1 in enumerate(reasonings):
                for idx2,reasoning2 in enumerate(reasonings):
                    if idx1 >= idx2:
                        continue


                    configs1 = reasoning_dict[reasoning1]
                    configs2 = reasoning_dict[reasoning2]

                    for config1, config2 in zip(configs1, configs2):

                        sigval        = results_dict[(config1, config2)][2]

                        if sigval == 0:
                            hypothesis_cnt_dict[f'{reasoning1} vs {reasoning2}'][task][dimension]['equal'] += 1
                            hypothesis_cnt_dict[f'{reasoning1} vs {reasoning2}'][task]['All']['equal'] += 1
                        
                        else:
                            if results_dict[(config1, config2)][0] == config1:
                                hypothesis_cnt_dict[f'{reasoning1} vs {reasoning2}'][task][dimension]['better'] += 1
                                hypothesis_cnt_dict[f'{reasoning1} vs {reasoning2}'][task]['All']['better'] += 1

                            elif results_dict[(config1, config2)][0] == config2:

                                hypothesis_cnt_dict[f'{reasoning1} vs {reasoning2}'][task][dimension]['worse'] += 1
                                hypothesis_cnt_dict[f'{reasoning1} vs {reasoning2}'][task]['All']['worse'] += 1
                        

    # pprint(hypothesis_cnt_dict)

    dimension_dict = {
        'base': 'Base',
        'adversarial': 'Adv',
        'adversarial_prompt': 'Adv_P',
        'domain': 'T2C',
        'domain_vocab': 'Voc',
        'compositional': 'Comp',
        'length': 'Len',        
        'All':'All'
    }

    

    comp_dict = ddict(list)
    for hypothesis in hypothesis_cnt_dict:
        for task in hypothesis_cnt_dict[hypothesis]:
            for dimension in dimension_dict.keys():
                tot = hypothesis_cnt_dict[hypothesis][task][dimension]['better'] + hypothesis_cnt_dict[hypothesis][task][dimension]['worse'] + hypothesis_cnt_dict[hypothesis][task][dimension]['equal']

                # comp_dict['task'].append(task.capitalize()[0])
                # comp_dict['task'].append(task.capitalize()[0])
                # comp_dict['task'].append(task.capitalize()[0])
                comp_dict['task'].append(task)
                comp_dict['task'].append(task)
                comp_dict['task'].append(task)
                comp_dict['dim'].append(dimension_dict[dimension])
                comp_dict['dim'].append(dimension_dict[dimension])
                comp_dict['dim'].append(dimension_dict[dimension])
                comp_dict['hyp'].append(hypothesis)
                comp_dict['hyp'].append(hypothesis)
                comp_dict['hyp'].append(hypothesis)
                comp_dict['Performance'].append('Better')
                comp_dict["Fraction"].append(hypothesis_cnt_dict[hypothesis][task][dimension]['better']/tot)
                comp_dict['Performance'].append('Equal')
                comp_dict["Fraction"].append(hypothesis_cnt_dict[hypothesis][task][dimension]['equal']/tot)
                comp_dict['Performance'].append('Worse')
                comp_dict["Fraction"].append(hypothesis_cnt_dict[hypothesis][task][dimension]['worse']/tot)
            
    all_df = pd.DataFrame(comp_dict)

    ### remove Dense vs MOE cases ###

    comp_df = all_df[~all_df['hyp'].str.contains('Dense vs MOE')]

    time_df = comp_df[comp_df['task'] == 'time']

    sns.set(style="whitegrid")
    sns.catplot(time_df, row="hyp", y="Fraction", x="Performance", hue="Performance", kind="bar", col='dim', palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/all_hypothesis_time_validation.pdf', bbox_inches="tight")


    event_df = comp_df[comp_df['task'] == 'event']

    sns.set(style="whitegrid")
    sns.catplot(event_df, row="hyp", y="Fraction", x="Performance", hue="Performance", kind="bar", col='dim', palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/all_hypothesis_event_validation.pdf', bbox_inches="tight")


    ################ only look at Inductive configurations ##################
    
    for idx, row in comp_df.iterrows():
        icount = row['hyp'].count('I')
        if icount <2:
            comp_df = comp_df.drop(idx)

    time_df = comp_df[comp_df['task'] == 'time']
    ## check cases where col == 'hyp' has two I in it     
    sns.set(style="whitegrid")
    sns.catplot(time_df, row="hyp", col='dim', y="Fraction", x="Performance", hue="Performance", kind="bar", palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/only_inductive_hypothesis_time_validation.pdf', bbox_inches="tight")

    

    event_df = comp_df[comp_df['task'] == 'event']

    sns.set(style="whitegrid")
    sns.catplot(event_df,row="hyp", col='dim', y="Fraction", x="Performance", hue="Performance", kind="bar", palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/only_inductive_hypothesis_event_validation.pdf', bbox_inches="tight")


    curr_df = comp_df[comp_df['dim']=='All']

    curr_df['task']= curr_df['task'].apply(lambda x: x.capitalize()[0])

    sns.set(style="whitegrid")
    sns.catplot(curr_df,row="task", col='hyp', y="Fraction", x="Performance", hue="Performance", kind="bar", palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/only_inductive_all_hypothesis_tasks_validation.pdf', bbox_inches="tight")

    ##### check the Dense vs MOE hypothesis validation #####

    curr_df = all_df[all_df['hyp'].str.contains('Dense vs MOE')]
    curr_df['task']= curr_df['task'].apply(lambda x: x.capitalize()[0])

    sns.set(style="whitegrid")
    sns.catplot(curr_df,row="task", col='dim', y="Fraction", x="Performance", hue="Performance", kind="bar", palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/Dense_vs_MOE_hypothesis_tasks_validation.pdf', bbox_inches="tight")


def compute_perchange():

    dimensions       =  ['length', 'adversarial', 'adversarial_prompt', 'domain', 'compositional', 'domain_base', 'domain_clinical', 'domain_vocab']
    tasks            =  ['time', 'event']

    sig_diff_dict = ddict(list)

    dimension_dict = {
        'base': 'Base',
        'length': 'Len',
        'adversarial': 'Adv',
        'adversarial_prompt': 'Adv_P',
        'domain': 'T2C',
        'compositional': 'Comp',
        'domain_base': 'C2T',
        'domain_clinical': 'Clin',
        'domain_vocab': 'Voc',
        'All':'All'
    }

    for task in tasks:

        base_sig_df = pd.read_csv(f'../analysis_results/sigtest/all_dim_results_{task}_base_sigtest.csv')

        ## find out the model pairs that are significant in the base setting ##

        base_model_pairs            = set()
        base_notworse_model_pairs   = {'better': set(), 'equal': set()}

        for idx, row in base_sig_df.iterrows():
            key1, key2  = row['key1'], row['key2']
            m1, m2      = get_model_name(key1), get_model_name(key2)

            if row['p_val'] < 0.05:
                if row['better_model'] == key1:
                    base_model_pairs.add((m1, m2))
                    base_notworse_model_pairs['better'].add((m1, m2))
                else:
                    base_model_pairs.add((m2, m1))
                    base_notworse_model_pairs['better'].add((m2, m1))
            
            elif row['p_val'] >= 0.05:
                base_notworse_model_pairs['equal'].add((m1, m2))
                base_notworse_model_pairs['equal'].add((m2, m1))

        
        for dimension in dimensions:

            curr_df = pd.read_csv(f'../analysis_results/sigtest/all_dim_results_{task}_{dimension}_sigtest.csv')

            curr_model_pairs                = set()
            curr_notworse_model_pairs       = {'better': set(), 'equal': set()}
            

            for idx, row in curr_df.iterrows():
                key1, key2  = row['key1'], row['key2']
                m1, m2      = get_model_name(key1), get_model_name(key2)

                if row['p_val'] < 0.05:
                    if row['better_model'] == key1:
                        curr_model_pairs.add((m1, m2))                        
                        curr_notworse_model_pairs['better'].add((m1, m2))
                    else:
                        curr_model_pairs.add((m2, m1))
                        curr_notworse_model_pairs['better'].add((m2, m1))
                
                elif row['p_val'] >= 0.05:
                    curr_notworse_model_pairs['equal'].add((m1, m2))
                    curr_notworse_model_pairs['equal'].add((m2, m1))
                    

            same_sig_cnt = len(base_model_pairs.intersection(curr_model_pairs))
            same_sig_cnt2 = len(base_notworse_model_pairs['better'].intersection(curr_notworse_model_pairs['better'])) + len(base_notworse_model_pairs['equal'].intersection(curr_notworse_model_pairs['equal']))//2

            sig_diff_dict['sig'].append('M1 is better than M2')
            sig_diff_dict['task'].append(task)
            sig_diff_dict['dimension'].append(dimension_dict[dimension])
            sig_diff_dict['probability'].append(same_sig_cnt/len(base_model_pairs))

            sig_diff_dict['sig'].append('M2 is not worse than M1')
            sig_diff_dict['task'].append(task)
            sig_diff_dict['dimension'].append(dimension_dict[dimension])
            sig_diff_dict['probability'].append(same_sig_cnt2/(len(base_notworse_model_pairs['better'])+ len(base_notworse_model_pairs['equal'])//2))


    sig_diff_df = pd.DataFrame(sig_diff_dict)
    sns.set(style="whitegrid")
    sns.catplot(sig_diff_df, y='dimension', x='probability', col='task', kind='bar', row ='sig')
    plt.ylabel('Different Generalization Dimensions')
    plt.savefig('../visualizations/sig_diff.pdf', bbox_inches="tight")




def model_size_exps():

    tasks      = ['timex', 'event']     
    dimensions = ['base', 'length', 'adversarial', 'domain', 'compositional']
    sizes      = [3, 24, 123]

    results_dict = ddict(list)

    task_map   = {
        'timex': 'time',
        'event': 'event'
    }

    dimension_dict = {
        'base': 'Base',
        'length': 'Len',
        'adversarial': 'Adv',
        'domain': 'T2C',
        'compositional': 'Comp',
    }


    for task in tasks:
        for dimension in dimensions:
            for size in sizes:
                for setting in ['abductive', 'deductive', 'inductive', 'deductive+abductive', 'inductive+abductive', 'inductive+deductive' , 'inductive+deductive+abductive']:

                    if dimension == 'compositional' and task == 'event':
                        fpath = f'/data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/model-size-experiments/timex_mistral{size}b_{dimension}/timex_mistral{size}b_{setting}_{dimension}.jsonl'
                    else:
                        fpath = f'/data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/model-size-experiments/{task}_mistral{size}b_{dimension}/{task}_mistral{size}b_{setting}_{dimension}.jsonl'

                    try:
                        res = json.load(open(fpath, 'r'))
                        f1 = res['results'][0]['groups']['hungarian']['word']['f1']
                        results_dict['task'].append(task_map[task])
                        results_dict['dimension'].append(dimension)
                        results_dict['size'].append(size)
                        results_dict['setting'].append(setting)
                        results_dict['f1'].append(f1)

                        # print(f'Finished for {task} with size {size} and dimension {dimension} and setting {setting} with f1 {f1}')
                    except Exception as e:
                        print(e)
                        continue
                        

    
    for task in ['time', 'event']:

        for dimension in dimensions:

            for setting in ['abductive', 'deductive', 'inductive', 'deductive+abductive', 'inductive+abductive', 'inductive+deductive' , 'inductive+deductive+abductive']:

                if dimension == 'compositional' and task == 'event':
                    fpath = f'/data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/timex_results/timex_{dimension}/timex_mistral8b_{dimension}/timex_mistral8b_{setting}_{dimension}.jsonl'
                else:
                    fpath = f'/data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/{task}x_results/{task}x_{dimension}/{task}x_mistral8b_{dimension}/{task}x_mistral8b_{setting}_{dimension}.jsonl'

                try:
                    res = json.load(open(fpath, 'r'))
                    f1 = res['results'][0]['groups']['hungarian']['word']['f1']
                    results_dict['task'].append(task)
                    results_dict['dimension'].append(dimension)
                    results_dict['size'].append(8)
                    results_dict['setting'].append(setting)
                    results_dict['f1'].append(f1)
                    print(f'Finished for {task} with size 8 B and dimension {dimension} and setting {setting} with f1 {f1}')

                except Exception as e:
                    print(e)


                    import pdb; pdb.set_trace()



    results_df = pd.DataFrame(results_dict)

    # change the dimensions with respect to the dimension dict

    results_df['dimension'] = results_df['dimension'].apply(lambda x: dimension_dict[x])

    sns.set(style="darkgrid")
    # sns.set_context("paper", font_scale=1.1)
    
    g = sns.relplot(
        data=results_df,
        x="size",
        y="f1",
        col="task",
        hue="dimension",
        kind="line",
        marker="o",
        dashes=False,
        errorbar=None,        # removes error shades
    )

    # log scale for x-axis
    g.set(xscale="log")

    # highlight only specific x-axis points
    ticks = [3, 8, 24, 123]
    tick_lbls = ['3B', '8B', '24B', '123B']

    for ax in g.axes.flat:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_lbls)
        ax.set_xlim(2.5, 130)

    # set legend outside the plot and above the plot and make the legend horizontal 

    # g._legend.set_bbox_to_anchor((0.5, 1.15))

    sns.move_legend(g, "lower center", ncol=5, bbox_to_anchor=(.5, 1.01))

    

    plt.savefig('../visualizations/model_size_experiments_noerr_bars.pdf', bbox_inches="tight")
    plt.close('all')
    plt.clf()




def validate_ft_hyps():

    models            = ['t5', 'roberta', 'gpt2']

    dimensions       =  ['base', 'length', 'adversarial', 'adversarial_prompt', 'domain', 'domain_vocab']
    tasks            =  ['time', 'event']

    config_dict      = {'t5': 'T5', 'roberta': 'RoBERTa', 'gpt2': 'GPT-2'}

    hypothesis_cnt_dict = ddict(lambda: ddict(lambda: ddict(int)))    

    for task in tasks:
        for dimension in dimensions:

            results_dict                = {}
            sigtest_df                  = pd.read_csv(f'../analysis_results/sigtest/finetuned_all_dim_results_{task}_{dimension}_sigtest.csv')

            for idx, row in sigtest_df.iterrows():
                m1, m2  = row['key1'], row['key2']

                if row['better_model'] == m1:
                    results_dict[(m1,m2)] = (m1, m2, 1 if row['p_val'] <= 0.05 else 0)
                    results_dict[(m2,m1)] = (m1, m2, 1 if row['p_val'] <= 0.05 else 0)

                elif row['better_model'] == m2:
                    results_dict[(m1,m2)] = (m2, m1, 1 if row['p_val'] <= 0.05 else 0)
                    results_dict[(m2,m1)] = (m2, m1, 1 if row['p_val'] <= 0.05 else 0)

            print(len(results_dict))             
                
            for idx1, m1 in enumerate(models):
                for idx2,m2 in enumerate(models):
                    if idx1 >= idx2:
                        continue

                    config1, config2    = m1, m2
                    sigval              = results_dict[(config1, config2)][2]

                    if sigval == 0:
                        hypothesis_cnt_dict[f'{config_dict[m1]} vs {config_dict[m2]}'][task]['equal'] += 1
                    
                    else:
                        if results_dict[(config1, config2)][0] == config1:
                            hypothesis_cnt_dict[f'{config_dict[m1]} vs {config_dict[m2]}'][task]['better'] += 1
                            
                        elif results_dict[(config1, config2)][0] == config2:
                            hypothesis_cnt_dict[f'{config_dict[m1]} vs {config_dict[m2]}'][task]['worse'] += 1
                            
                    

    # pprint(hypothesis_cnt_dict)

    dimension_dict = {
        'base': 'Base',
        'length': 'Len',
        'adversarial': 'Adv',
        'adversarial_prompt': 'Adv_P',
        'domain': 'T2C',
        'domain_vocab': 'Voc',
        'All':'All'
    }

    comp_dict = ddict(list)

    for hypothesis in hypothesis_cnt_dict:
        for task in hypothesis_cnt_dict[hypothesis]:

            tot = hypothesis_cnt_dict[hypothesis][task]['better'] + hypothesis_cnt_dict[hypothesis][task]['worse'] + hypothesis_cnt_dict[hypothesis][task]['equal']

            comp_dict['task'].append(task)
            comp_dict['task'].append(task)
            comp_dict['task'].append(task)
            comp_dict['hyp'].append(hypothesis)
            comp_dict['hyp'].append(hypothesis)
            comp_dict['hyp'].append(hypothesis)
            comp_dict['hue'].append('Better')
            comp_dict["Fraction"].append(hypothesis_cnt_dict[hypothesis][task]['better']/tot)
            comp_dict['hue'].append('Equal')
            comp_dict["Fraction"].append(hypothesis_cnt_dict[hypothesis][task]['equal']/tot)
            comp_dict['hue'].append('Worse')
            comp_dict["Fraction"].append(hypothesis_cnt_dict[hypothesis][task]['worse']/tot)
            
    comp_df = pd.DataFrame(comp_dict)

    time_df = comp_df[comp_df['task'] == 'time']

    sns.set(style="whitegrid")
    sns.catplot(time_df, col="hyp", y="Fraction", x="hue", hue="hue", kind="bar", palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/finetuned_hypothesis_time_validation.pdf', bbox_inches="tight")


    event_df = comp_df[comp_df['task'] == 'event']

    sns.set(style="whitegrid")
    sns.catplot(event_df,col="hyp", y="Fraction", x="hue", hue="hue", kind="bar", palette={'Better': 'green', 'Equal': 'yellow', 'Worse': 'red'}, height=3.0)
    plt.ylabel('Proportion')
    plt.savefig('../visualizations/finetuned_hypothesis_event_validation.pdf', bbox_inches="tight")

    ################ only look at Inductive configurations ##################

    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute significance test')
    parser.add_argument('--task',       type=str, default='timex', help='task')
    parser.add_argument('--step',       type=str, default='analyse', required=True)
    
    args        =  parser.parse_args()

    if args.step == 'val_hyps':
        validate_hyps()
   
    if args.step == 'perchange':
        compute_perchange()
    
    if args.step == 'size':
        model_size_exps()
    
    if args.step == 'ft_val_hyps':
        print('Doing hypothesis validation')

        validate_ft_hyps()
