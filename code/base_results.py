from eval_script import *
import os, json
from collections import defaultdict as ddict
from glob import glob
import pandas as pd
import random
import argparse

def create_base_results():

    models              = ['llama8b', 'llama17b', 'mistral8b', 'mistral13b', 'qwen7b', 'qwen22b']

    dimensions          = ['domain_vocab', 'base', 'length', 'adversarial', 'domain', 'compositional', 'domain_base', 'domain_clinical', 'adversarial_prompt']

    reasoning_types     = ['abductive', 'deductive', 'inductive', 'deductive+abductive', 'inductive+abductive', 'inductive+deductive' , 'inductive+deductive+abductive']
    
    time_datas          = ['time','event']

    results_dict        = ddict(lambda: ddict(lambda: ddict(lambda: ddict(dict))))

    agg_results_dict    = ddict(list)

    start_dir = f'../../mdg_git/results/updated_outputs/'

    for org_time_data in time_datas:

        for dimension in dimensions:

            if dimension == 'compositional' and org_time_data == 'event':
                time_data = 'time'
            else:
                time_data = org_time_data

            for model in models:
                for reasoning_type in reasoning_types:

                    if dimension == 'domain_base' or dimension == 'domain_clinical':
                        result_path = f'{start_dir}/{time_data}x_results/{time_data}x_{dimension}/{time_data}x_{model}_{dimension}/{time_data}x_{model}_{dimension}_output/{time_data}x_{model}_{reasoning_type}_domain.jsonl'
                    
                    elif dimension == 'adversarial_prompt':

                        result_path = f'{start_dir}/{time_data}x_results/{time_data}x_{dimension}/{time_data}x_{model}_adversarial/{time_data}x_{model}_adversarial_output/{time_data}x_{model}_{reasoning_type}_adversarial.jsonl'
                    
                    elif dimension == 'domain_vocab':

                        # /data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/timex_results/timex_domain_vocab/timex_llama17b_domain
                        result_path = f'{start_dir}/{time_data}x_results/{time_data}x_{dimension}/{time_data}x_{model}_domain/{time_data}x_{model}_domain_output/{time_data}x_{model}_{reasoning_type}_domain.jsonl'

                    else:
                        result_path = f'{start_dir}/{time_data}x_results/{time_data}x_{dimension}/{time_data}x_{model}_{dimension}/{time_data}x_{model}_{dimension}_output/{time_data}x_{model}_{reasoning_type}_{dimension}.jsonl'

                    golds, preds, texts = [], [], []

                    try:
                        lines =  open(result_path).readlines()
                    except Exception as e:
                        print(f'\nError reading file in {result_path}:\n')
                        
                    for line in lines:
                        elem = json.loads(line)

                        key  = (elem['doc_id'], elem['text'])

                        if dimension == 'compositional':
                            elem[f'gold_{time_data}_expressions']           = elem['gold_expressions']
                            elem[f'post_processed_{time_data}_expressions'] = elem['post_processed_expressions']
                        
                        if 'gold' not in results_dict[time_data][dimension][key]:
                            results_dict[time_data][dimension][key]['gold'] = elem[f'gold_{time_data}_expressions']
                        else:
                            gold_expressions = results_dict[time_data][dimension][key]['gold']

                            assert set(gold_expressions) == set(elem[f'gold_{time_data}_expressions']), f"Gold expressions do not match for doc_id: {elem['doc_id']} and text: {elem['text']}"

                        model_key = f'{model}_{reasoning_type}'

                        results_dict[time_data][dimension][key][model_key] = elem[f'post_processed_{time_data}_expressions']
                        
                        golds.append(elem[f'gold_{time_data}_expressions'])
                        preds.append(elem[f'post_processed_{time_data}_expressions'])
                    
                    mf1 = evaluate_partial_and_match(golds, preds)['word_f1_match']

                    instancewise_scores = instancewise_metrics(golds, preds)['instancewise_match_word']

                    sum_prec, sum_rec, total_preds, total_golds = 0, 0, 0, 0

                    # model_value_key = (model, reasoning_type, 'value')

                    lines =  open(result_path).readlines()
                    
                    for line, inst in zip(lines, instancewise_scores):
                        elem = json.loads(line)
                        key  = (elem['doc_id'], elem['text'])

                        sum_prec += inst['sum_prec']
                        sum_rec += inst['sum_rec']
                        total_preds += inst['nP']
                        total_golds += inst['nG']

                        results_dict[time_data][dimension][key][f'{model}_{reasoning_type}_prec'] = inst['sum_prec']
                        results_dict[time_data][dimension][key][f'{model}_{reasoning_type}_rec'] = inst['sum_rec']
                        results_dict[time_data][dimension][key][f'{model}_{reasoning_type}_nP'] = inst['nP']
                        results_dict[time_data][dimension][key][f'{model}_{reasoning_type}_nG'] = inst['nG']

                    
                    precision = (sum_prec / total_preds) if total_preds > 0 else 0.0
                    recall    = (sum_rec  / total_golds) if total_golds > 0 else 0.0
                    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

                    assert f1 == mf1, f"F1 scores do not match for model: {model}, reasoning_type: {reasoning_type}, dimension: {dimension}, time_data: {time_data}"
                    assert len(instancewise_scores) == len(golds) == len(preds), f"Instancewise scores length does not match golds and preds length for model: {model}, reasoning_type: {reasoning_type}, dimension: {dimension}, time_data: {time_data}"

                    print(f'Model: {model}, Reasoning Type: {reasoning_type}, Dimension: {dimension}, Time Data: {time_data}, Lines: {len(golds)} Word F1 Match: {mf1:.4f}')

                    agg_results_dict['task'].append(org_time_data)
                    agg_results_dict['dimension'].append(dimension)
                    agg_results_dict['model'].append(model)
                    agg_results_dict['reasoning_type'].append(reasoning_type)
                    agg_results_dict['precision'].append(precision)
                    agg_results_dict['recall'].append(recall)
                    agg_results_dict['f1'].append(f1)
    
    agg_results_df = pd.DataFrame(agg_results_dict)
    agg_results_df.to_csv('../analysis_results/agg_results.csv', index=False)

    base_results_dict = ddict(list)
    for time_data in results_dict:
        for dimension in results_dict[time_data]:
            for key in results_dict[time_data][dimension]:
                gold_expressions = results_dict[time_data][dimension][key]['gold']
                base_results_dict['task'].append(time_data)
                base_results_dict['dimension'].append(dimension)
                base_results_dict['key'].append(key)
                for model_key in results_dict[time_data][dimension][key]:
                    if model_key == 'gold':
                        continue
                    value = results_dict[time_data][dimension][key][model_key]
                    base_results_dict[model_key].append(value)

    for key in base_results_dict:
        print(f'{key}: {len(base_results_dict[key])}')
    
    # import pdb; pdb.set_trace()
    base_results_df = pd.DataFrame(base_results_dict)
    base_results_df.to_csv('../analysis_results/all_dim_results.csv', index=False)




def compute_bootstrapped_df(df, model1, model2, num_iter = 100, ft_flag = True):
    
    # agg_scores(df, model1, model2)
    if not ft_flag:
        model1_f1, model2_f1 = agg_scores(df, model1, model2)
    else:
        model1_f1, model2_f1 = agg_ft_scores(df, model1, model2)
    

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

        curr_df         = df.iloc[curr_idx_list]

        if not ft_flag:
            curr_model1_f1, curr_model2_f1 = agg_scores(curr_df, model1, model2)
        else:
            curr_model1_f1, curr_model2_f1 = agg_ft_scores(curr_df, model1, model2)
        

        # curr_model1_f1, curr_model2_f1 = agg_scores(curr_df, model1, model2)
        
        curr_diff        = curr_model1_f1 - curr_model2_f1
        if curr_diff > 2*sys_diff:
            cnt += 1
            
    
    sig_val = ''
    p_val = cnt/num_iter

    if p_val < 0.001 :
        sig_val = '***'
    elif p_val < 0.01: 
        sig_val = '**'
    elif p_val < 0.05:
        sig_val = '*'
    
    results_dict = {
        'p_val': p_val, 
        'sig_val': sig_val,
        'better_model': model1,
        'better_f1': round(model1_f1,4),
        'worse_model': model2,
        'worse_f1': round(model2_f1,4)
    }
    
    return results_dict



def compare_results():
    df                = pd.read_csv('../analysis_results/all_dim_results.csv')

    models            = ['llama8b', 'llama17b', 'mistral7b', 'mistral13b', 'qwen7b', 'qwen22b']

    reasoning_types   = ['abductive', 'deductive', 'inductive', 'deductive+abductive', 'inductive+abductive', 'inductive+deductive' , 'inductive+deductive+abductive']
    
    time_datas        = ['event', 'time']

    dimensions        = ['base', 'length', 'adversarial', 'domain']

    keys = []
    for model in models:
        for reasoning_type in reasoning_types:
            if model == 'llama8b' and reasoning_type == 'inductive':
                continue
            model_key = (model, reasoning_type)
            keys.append(model_key)

    tot_cnt     = 0
    done_cases  = []    

    if os.path.exists('../analysis_results/all_dim_results_sigtest.csv'):
        sigtest_df = pd.read_csv('../analysis_results/all_dim_results_sigtest.csv')
        for idx, row in sigtest_df.iterrows():
            done_cases.append((row['task'], row['dimension'], row['key1'], row['key2']))
    
    sigtest_dict = ddict(list)

    for time_data in time_datas:

        for dimension in dimensions:

            curr_df = df[(df['task'] == time_data)& (df['dimension'] == dimension)]
            for idx1, key1 in enumerate(keys):
                for idx2, key2 in enumerate(keys):
                    if idx2 <= idx1:
                        continue

                    if (time_data, dimension, key1, key2) in done_cases:
                        continue
                    
                    try:
                        sigtest_results = compute_bootstrapped_df(curr_df, key1, key2)
                        sigtest_dict['task'].append(time_data)
                        sigtest_dict['dimension'].append(dimension)
                        sigtest_dict['key1'].append(key1)
                        sigtest_dict['key2'].append(key2)

                        for key in sigtest_results:
                            sigtest_dict[key].append(sigtest_results[key])
                    except Exception as e:
                        print(f'Error for {key1} and {key2} for {time_data}: {e}')
                        continue

                    tot_cnt +=1

                    if tot_cnt % 10 == 0:
                        print(f'Done for {tot_cnt} comparisons', end='\r')
                        temp_df = pd.DataFrame(sigtest_dict)

                        if len(done_cases) == 0:
                            sigtest_df = temp_df
                        else:
                            sigtest_df = pd.concat([pd.read_csv('../analysis_results/all_dim_results_sigtest.csv'), temp_df], ignore_index=True)
                        
                        done_cases.append([(row['task'], row['dimension'], row['key1'], row['key2']) for idx, row in temp_df.iterrows()])
                        sigtest_df.to_csv('../analysis_results/all_dim_results_sigtest.csv', index=False)
                        sigtest_dict = ddict(list)


                    

    # sigtest_df = pd.DataFrame(sigtest_dict)
    # sigtest_df.to_csv('../analysis_results/base_results_sigtest.csv', index=False)




def compare_sigtest_results():

    # dimensions       =  ['base', 'length', 'adversarial', 'domain', 'compositional', 'domain_base', 'domain_clinical','adversarial_prompt', 'domain_vocab']
    dimensions       = ['domain_base', 'adversarial_prompt']
    tasks            =  ['time', 'event']

    df = pd.read_csv('../analysis_results/all_dim_results.csv')

    models            = ['llama8b', 'llama17b', 'mistral8b', 'mistral13b', 'qwen7b', 'qwen22b']

    reasoning_types   = ['abductive', 'deductive', 'inductive', 'deductive+abductive', 'inductive+abductive', 'inductive+deductive' , 'inductive+deductive+abductive']

    for task in tasks:
        for dimension in dimensions:

            if dimension == 'compositional' and task == 'event':
                continue

            # if os.path.exists(f'../analysis_results/sigtest/all_dim_results_{task}_{dimension}_sigtest.csv'):

            #     if len(pd.read_csv(f'../analysis_results/sigtest/all_dim_results_{task}_{dimension}_sigtest.csv')) >= 861:
            #         continue

            #     print(f'Sigtest results already exist for {task} and {dimension}, skipping...')

            keys = []
            for model in models:
                for reasoning_type in reasoning_types:
                    model_key = f'{model}_{reasoning_type}'
                    keys.append(model_key)

            print('Number of unique keys:', len(keys))

            tot_cnt     = 0
            done_cases  = []    

            sigtest_dict = ddict(list)

            curr_df = df[(df['task'] == task)& (df['dimension'] == dimension)]

            for idx1, key1 in enumerate(keys):
                for idx2, key2 in enumerate(keys):

                    if idx2 >= idx1:
                        continue

                    try:
                        sigtest_results = compute_bootstrapped_df(curr_df, key1, key2)
                        sigtest_dict['task'].append(task)
                        sigtest_dict['dimension'].append(dimension)
                        sigtest_dict['key1'].append(key1)
                        sigtest_dict['key2'].append(key2)

                        for key in sigtest_results:
                            sigtest_dict[key].append(sigtest_results[key])
                    except Exception as e:
                        print(f'Error for {key1} and {key2} for {task} and {dimension}')

                    tot_cnt +=1

                    print(f'Done for {tot_cnt} comparisons for {task} and {dimension}', end='\r')

            sigtest_df = pd.DataFrame(sigtest_dict)
            sigtest_df.to_csv(f'../analysis_results/sigtest/all_dim_results_{task}_{dimension}_sigtest.csv', index=False)

            print(f'\nLength of the sigtest results for {task} and {dimension}: {len(sigtest_dict)}\n')



def create_finetuned_results():

    dimensions          = ['base', 'length', 'adversarial', 'adversarial-prompt-based', 'domain', 'domain-vocab']

    models              = ['gpt2',  't5',  'roberta']    

    tasks               = ['time', 'event']

    results_dict        = ddict(lambda: ddict(lambda: ddict(lambda: ddict(dict))))

    agg_results_dict    = ddict(list)

    for task in tasks:
        for dimension in dimensions:
            print('--------------------------------------------------------------------------------------------------')
            for model in models:

                golds, preds, texts = [], [], []

                lines = open(f'/data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/fine-tuned-results/{task}x/{model}/{task}x-{dimension}-{model}.jsonl').readlines()
                                    
                for line in lines:
                    elem = json.loads(line)

                    if 'text' in elem:
                        key  = elem['text']
                    else:
                        continue

                    if task == 'event':
                        gold = elem['gold_event_expressions']
                        pred = elem['post_processed_event_expressions']
                    
                    if task == 'time':
                        gold = elem['gold_timex_expressions']
                        pred = elem['post_processed_timex_expressions']
                    
                
                    if 'gold' not in results_dict[task][dimension][key]:
                        results_dict[task][dimension][key]['gold'] = gold
                    else:
                        gold = results_dict[task][dimension][key]['gold']

                        
                    model_key = f'{model}'

                    results_dict[task][dimension][key][model_key] = pred
                    
                    golds.append(gold)
                    preds.append(pred)
                
                mf1 = evaluate_partial_and_match(golds, preds)['word_f1_match']

                instancewise_scores = instancewise_metrics(golds, preds)['instancewise_match_word']

                sum_prec, sum_rec, total_preds, total_golds = 0, 0, 0, 0

                # model_value_key = (model, reasoning_type, 'value')

                lines = open(f'/data/shire/projects/generalization/temporal_generalization/mdg_git/results/updated_outputs/fine-tuned-results/{task}x/{model}/{task}x-{dimension}-{model}.jsonl').readlines()
                                
                for line, inst in zip(lines, instancewise_scores):
                    elem = json.loads(line)
                    key  = elem['text']

                    sum_prec += inst['sum_prec']
                    sum_rec += inst['sum_rec']
                    total_preds += inst['nP']
                    total_golds += inst['nG']

                    results_dict[task][dimension][key][f'{model}_prec'] = inst['sum_prec']
                    results_dict[task][dimension][key][f'{model}_rec'] = inst['sum_rec']
                    results_dict[task][dimension][key][f'{model}_nP'] = inst['nP']
                    results_dict[task][dimension][key][f'{model}_nG'] = inst['nG']
                    results_dict[task][dimension][key][f'{model}'] = 1
                
                precision = (sum_prec / total_preds) if total_preds > 0 else 0.0
                recall    = (sum_rec  / total_golds) if total_golds > 0 else 0.0
                f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

                assert f1 == mf1, f"F1 scores do not match for model: {model}, dimension: {dimension}, task: {task}"

                assert len(instancewise_scores) == len(golds) == len(preds), f"Instancewise scores length does not match golds and preds length for model: {model}, dimension: {dimension}, task: {task}"

                print(f'Model: {model},  Dimension: {dimension}, Task: {task}, Lines: {len(golds)} Word F1 Match: {mf1:.4f}')

                agg_results_dict['task'].append(task)
                agg_results_dict['dimension'].append(dimension)
                agg_results_dict['model'].append(model)
                agg_results_dict['precision'].append(precision)
                agg_results_dict['recall'].append(recall)
                agg_results_dict['f1'].append(f1)

    common_keys = ddict(lambda: ddict(set))

    for time_data in results_dict:
        for dimension in results_dict[time_data]:
            for key in results_dict[time_data][dimension]:
                gflag = results_dict[time_data][dimension][key]['gpt2'] if 'gpt2' in results_dict[time_data][dimension][key] else 0
                rflag = results_dict[time_data][dimension][key]['roberta'] if 'roberta' in results_dict[time_data][dimension][key] else 0
                tflag = results_dict[time_data][dimension][key]['t5'] if 't5' in results_dict[time_data][dimension][key] else 0

                if gflag == 1 and rflag == 1 and tflag == 1:
                    common_keys[time_data][dimension].add(key)

    agg_results_df = pd.DataFrame(agg_results_dict)
    agg_results_df.to_csv('../analysis_results/finetuned_agg_results.csv', index=False)

    base_results_dict = ddict(list)
    for time_data in results_dict:
        for dimension in results_dict[time_data]:
            for key in list(common_keys[time_data][dimension]):
                gold_expressions = results_dict[time_data][dimension][key]['gold']
                base_results_dict['task'].append(time_data)
                base_results_dict['dimension'].append(dimension)
                base_results_dict['key'].append(key)
                for model_key in results_dict[time_data][dimension][key]:
                    if model_key == 'gold':
                        continue
                    value = results_dict[time_data][dimension][key][model_key]
                    base_results_dict[model_key].append(value)

    for key in base_results_dict:
        print(f'{key}: {len(base_results_dict[key])}')
    
    # import pdb; pdb.set_trace()
    base_results_df = pd.DataFrame(base_results_dict)
    base_results_df.to_csv('../analysis_results/finetuned_all_results.csv', index=False)
            




def compare_finetuned_sigtest_results():

    dimensions          = ['base', 'length', 'adversarial', 'adversarial-prompt-based', 'domain', 'domain-vocab']

    keys                = ['gpt2',  't5',  'roberta']    

    tasks               = ['time', 'event']

    df                  = pd.read_csv('../analysis_results/finetuned_all_results.csv')

    dimension_map = {'adversarial-prompt-based': 'adversarial_prompt', 'domain-vocab': 'domain_vocab', 'base': 'base', 'length': 'length', 'adversarial': 'adversarial', 'domain': 'domain'}

    for task in tasks:
        for dimension in dimensions:

            tot_cnt     = 0
            done_cases  = []    

            sigtest_dict = ddict(list)

            curr_df = df[(df['task'] == task)& (df['dimension'] == dimension)]

            for idx1, key1 in enumerate(keys):
                for idx2, key2 in enumerate(keys):

                    if idx2 >= idx1:
                        continue

                    try:
                        sigtest_results = compute_bootstrapped_df(curr_df, key1, key2, ft_flag=True)
                        sigtest_dict['task'].append(task)
                        sigtest_dict['dimension'].append(dimension_map[dimension])
                        sigtest_dict['key1'].append(key1)
                        sigtest_dict['key2'].append(key2)

                        for key in sigtest_results:
                            sigtest_dict[key].append(sigtest_results[key])
                    except Exception as e:
                        print(e)
                        print(f'Error for {key1} and {key2} for {task} and {dimension}')
                        # import pdb; pdb.set_trace()

                    tot_cnt +=1

                    print(f'Done for {tot_cnt} comparisons for {task} and {dimension}', end='\r')

            sigtest_df = pd.DataFrame(sigtest_dict)
            sigtest_df.to_csv(f'../analysis_results/sigtest/finetuned_all_dim_results_{task}_{dimension_map[dimension]}_sigtest.csv', index=False)

            print(f'\nLength of the sigtest results for {task} and {dimension}: {len(sigtest_dict)}\n')




if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Compute base results and significance tests for temporal generalization experiments.')
    parser.add_argument('--task', help='Choose either eventx or timex extraction task')
    parser.add_argument('--dimension', help='Choose either base, length, adversarial or domain dimension')
    parser.add_argument('--step', help="Choice of the action to perform: ")
    args = parser.parse_args()

    if args.step == 'create_base':
        create_base_results()

    if args.step == 'sigtest':
        compare_sigtest_results()
    
    if args.step == 'finetune':
        create_finetuned_results()
    
    if args.step == 'ft_sigtest':
        compare_finetuned_sigtest_results()

