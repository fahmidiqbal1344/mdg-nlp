import pdb
import spacy
from datasets import load_dataset
import argparse
import json
from collections import defaultdict as ddict
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

def extract_ner(args):

    if args.dataset == 'eventx':
        ds  = load_dataset('mdg-nlp/eventx-recognition-original')
    
        '''
        # features across different datasets
        features: ['text', 'time_expressions', 'event_expressions', 'signal_expressions', 'tlinks', 'sent_id', 'start_char', 'end_char', 'doc_id', 'dataset'],        
        '''
    
    elif args.dataset == 'timex':

        ds = load_dataset('mdg-nlp/timex-recognition-sentence-original')


    # Load the transformer-based English pipeline
    nlp = spacy.load("en_core_web_trf")

    splitwise_dict = {}

    for split in ['test', 'train', 'validation']:
        
        splitwise_dict[split] = []

        for idx, item in enumerate(ds[split]):
            instance_dict = {}

            text        = item['text']
            doc         = nlp(text)
            # Print entities: text, label, and character offsets
            instance_dict['doc_id']     = item['doc_id']
            instance_dict['sent_id']    = item['sent_id']
            instance_dict['text']       = item['text']
            instance_dict['ner']        = []

            for ent in doc.ents:    
                instance_dict['ner'].append({
                    'mention': ent.text,
                    'label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char
                })

            splitwise_dict[split].append(instance_dict)        

            print(f"Processed {idx+1}/{len(ds[split])} instances in {split} split.", end='\r')               

    with open(f'../data/{args.dataset}_ner.json', 'w') as f:
        json.dump(splitwise_dict, f, indent=4)


def analyse_ners(args):

    split    = 'test'
    ner_dict =  json.load(open(f'../data/{args.dataset}_ner.json'))[split]

    mention_cnt = ddict(lambda: ddict(int))
    sub_mention_cnt = ddict(int)

    imp_labels = ['NORP', 'ORG', 'PERSON', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LAW' 'LANGUAGE', 'PRODUCT', 'FAC']

    mention_dict        = ddict(list)
    sub_mention_dict    = ddict(list)

    banned_words = ['south', 'north', 'east', 'west', 'left', 'right', 'center', 'pro', 'anti']

    for instance in ner_dict:
        for ner in instance['ner']:
            
            lbl         = ner['label']
            mention     = ner['mention']

            if lbl in imp_labels:
                mention_cnt[lbl][mention] += 1

                for sub_mention in mention.split():

                    # check if sub_mention is a stop word or common noun and skip if it is
                    if sub_mention.lower() in STOP_WORDS:
                        continue

                    if sub_mention.lower() in banned_words:
                        continue

                    if '-' in sub_mention:
                        for sub_sub_mention in sub_mention.split('-'):
                            if sub_sub_mention.lower() in STOP_WORDS:
                                continue

                            if sub_sub_mention.lower() in banned_words:
                                continue

                            sub_mention_cnt[sub_sub_mention] += 1
                    else:
                        sub_mention_cnt[sub_mention] += 1
    
    for lbl in mention_cnt:
        print(f"Label: {lbl}\t Count {sum(mention_cnt[lbl].values())}")
    
    print('-----------------------------')
    
    for lbl in mention_cnt:
        for mention in mention_cnt[lbl]:
            mention_dict['label'].append(lbl)
            mention_dict['mention'].append(mention.strip())
            mention_dict['count'].append(mention_cnt[lbl][mention])
    
    for mention in sub_mention_cnt:
        sub_mention_dict['mention'].append(mention.strip())
        sub_mention_dict['count'].append(sub_mention_cnt[mention])

    mention_df = pd.DataFrame(mention_dict)
    mention_df.to_csv(f'../data/{args.dataset}_mentions.csv', index=False)

    sub_mention_df = pd.DataFrame(sub_mention_dict)
    sub_mention_df.to_csv(f'../data/{args.dataset}_sub_mentions.csv', index=False)


def substitute_eventx_mentions(args):

    split    = 'test'
    ner_dict =  json.load(open(f'../data/{args.dataset}_ner.json'))[split]

    substitute_mentions = pd.read_csv(f'../data/eventx_mention_substitutions.csv')

    eventx_mentions     = pd.read_csv(f'../data/{args.dataset}_mentions.csv')

    submention_substitute_dict     = {}

    mention_substitute_dict        = {}

    for idx, row in substitute_mentions.iterrows():
        assert len(row['mention']) == len(row['fictional'])
        submention_substitute_dict[row['mention']] = row['fictional']
    
    banned_mentions                 = ['pay', 'war', 'manned']

    for idx, row in eventx_mentions.iterrows():
        mention = row['mention']

        men_arr = []
        # how to replace all whitespace including \n and , with '-' using regex

        sub_mentions = mention.replace('-', '_').replace('\n','_').replace(' ', '_').split('_')

        for sub_mention in sub_mentions:
            
            if sub_mention.lower() in banned_mentions:
                men_arr.append(sub_mention)
            elif sub_mention in submention_substitute_dict:
                men_arr.append(submention_substitute_dict[sub_mention])
            else:
                men_arr.append(sub_mention)
        
        new_mention = ' '.join(men_arr)

        print(f'Done for {idx+1}/{len(eventx_mentions)} mentions.', end='\r')

        try:
            assert len(mention) == len(new_mention)
            eventx_mentions['length_issue'] = 0
        except:
            eventx_mentions['length_issue'] = 1

        mention_substitute_dict[mention] = new_mention
    
    eventx_mentions['fictional'] = eventx_mentions['mention'].apply(lambda x: mention_substitute_dict[x])
    eventx_mentions.to_csv(f'../data/{args.dataset}_mentions.csv', index=False)



def substitute_timex_mentions(args):

    split    = 'test'
    ner_dict =  json.load(open(f'../data/{args.dataset}_ner.json'))[split]

    substitute_mentions = pd.read_csv(f'../data/timex_submention_substitutions.csv')

    timex_mentions     = pd.read_csv(f'../data/{args.dataset}_mentions.csv')

    submention_substitute_dict     = {}

    mention_substitute_dict        = {}

    for idx, row in substitute_mentions.iterrows():
        assert len(row['mention']) == len(row['fictional'])
        submention_substitute_dict[row['mention']] = row['fictional']
    
    banned_mentions                 = ['pay', 'war', 'manned', 'new', 'year', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']

    for idx, row in timex_mentions.iterrows():
        mention = row['mention']

        men_arr = []
        # how to replace all whitespace including \n and , with '-' using regex

        sub_mentions = mention.replace('-', '_').replace('\n','_').replace(' ', '_').split('_')

        for sub_mention in sub_mentions:
            
            if sub_mention.lower() in banned_mentions:
                men_arr.append(sub_mention)
            elif sub_mention in submention_substitute_dict:
                men_arr.append(submention_substitute_dict[sub_mention])
            else:
                men_arr.append(sub_mention)
        
        new_mention = ' '.join(men_arr)

        print(f'Done for {idx+1}/{len(timex_mentions)} mentions.', end='\r')

        try:
            assert len(mention) == len(new_mention)
            timex_mentions['length_issue'] = 0
        except:
            timex_mentions['length_issue'] = 1

        mention_substitute_dict[mention] = new_mention
    
    timex_mentions['fictional'] = timex_mentions['mention'].apply(lambda x: mention_substitute_dict[x])
    timex_mentions.to_csv(f'../data/{args.dataset}_mentions.csv', index=False)



def update_data(args):

    split               = 'test'
    datax_mentions     = pd.read_csv(f'../data/{args.dataset}_mentions.csv')

    substitute_dict     = {}

    for idx, row in datax_mentions.iterrows():
        mention     = row['mention']
        fictional   = row['fictional']
        substitute_dict[(mention, row['label'])] = fictional
        print(f'Processed {idx+1}/{len(datax_mentions)} mentions.', end='\r')
        
    
    if args.dataset == 'eventx':
        ds = load_dataset('mdg-nlp/eventx-recognition-original')
    elif args.dataset == 'timex':
        ds = load_dataset('mdg-nlp/timex-recognition-sentence-original')


    ner_dict            =  json.load(open(f'../data/{args.dataset}_ner.json'))[split]

    new_texts           = []
    for item, inst in zip(ds[split], ner_dict):

        text        = item['text']
        doc_id      = item['doc_id']
        sent_id     = item['sent_id']

        assert doc_id == inst['doc_id'] and sent_id == inst['sent_id']

        for ner in inst['ner']:
            
            lbl         = ner['label']
            mention     = ner['mention']
            span        = (ner['start_char'], ner['end_char'])  
            if (mention, lbl) in substitute_dict:
                text    = text[:span[0]] + substitute_dict[(mention, lbl)] + text[span[1]:]
                assert len(text) == len(item['text'])

        # print(item['text'])    
        # print(text)
        # print('-----------------------------')
        

        if args.dataset == 'eventx':

            for event in item['event_expressions']:
                span = (event['start_char'], event['end_char'])
                event_text = text[span[0]:span[1]]
                try:
                    assert event_text == item['text'][span[0]:span[1]]
                except:
                    print(f"Event text mismatch for doc_id: {doc_id}, sent_id: {sent_id}")
                    print(f"Original event text: {item['text'][span[0]:span[1]]}")
                    print(f"Updated event text: {event_text}")
                    print('-----------------------------')
        
        if args.dataset == 'timex':

            for time in item['time_expressions']:
                span = (time['start_char'], time['end_char'])
                time_text = text[span[0]:span[1]]
                try:
                    assert time_text == item['text'][span[0]:span[1]]
                except:
                    if len(time_text) == len(item['text'][span[0]:span[1]]):
                        time['text'] = time_text                        

                    print(f"Time text mismatch for doc_id: {doc_id}, sent_id: {sent_id}")
                    print(f"Original time text: {item['text'][span[0]:span[1]]}")
                    print(f"Updated time text: {time_text}")
                    print('-----------------------------')

        new_texts.append(text)
    
    ds['test'] = ds['test'].add_column('domain_text', new_texts)
            
    print(f'Updated {idx+1}/{len(ds[split])} instances in {split} split.', end='\r')

    ds['test'].push_to_hub(f'mdg-nlp/{args.dataset}-recognition-sentence-vocab-substituted')

    

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='eventx', help='dataset name')
    args.add_argument('--step', type=str, default='create', help='step name')

    args = args.parse_args()

    if args.step == 'ner':
        extract_ner(args)
    
    if args.step == 'analyse':
        analyse_ners(args)
    
    if args.step == 'substitute':

        if args.dataset == 'eventx':
            substitute_eventx_mentions(args)
        elif args.dataset == 'timex':
            substitute_timex_mentions(args)
        
    
    if args.step == 'update':

        update_data(args)