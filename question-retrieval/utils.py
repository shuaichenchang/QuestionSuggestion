import numpy as np
import torch
import random
import json,json_lines
import os
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
from transformers import RobertaForTokenClassification
from transformers import AutoConfig, RobertaConfig
from transformers import RobertaTokenizer

def basic_statistics(args):
    with open(os.path.join(args.train_path, 'train.json'),'r') as f:
        train_set=json.load(f)
    with open(os.path.join(args.train_path, 'dev.json'),'r') as f:
        dev_set=json.load(f)
    with open(os.path.join(args.train_path, 'test.json'),'r') as f:
        test_set=json.load(f)
    print(len(train_set),len(dev_set),len(test_set))
    train_db=set()
    dev_db=set()
    test_db=set()
    for data in train_set:
        train_db.add(data['db_id'])
    for data in dev_set:
        dev_db.add(data['db_id'])
    for data in test_set:
        test_db.add(data['db_id'])
    print(len(train_db), len(dev_db-train_db), len(test_db-train_db))

def save_history_questions(args):

    history_questions=dict()
    with open(os.path.join(args.train_path, 'train.json'), 'r') as f:
        train_set = json.load(f)
        for item in train_set:
            if item['db_id'] not in history_questions:
                history_questions[item['db_id']]=[]
            history_questions[item['db_id']].append(item)
    with open(args.history_questions,"w") as f:
        json.dump(history_questions,f,indent=4)

    return history_questions

def load_history_questions(file_path):
    print("Load histories questions from:",file_path)
    with open(file_path,"r") as f:
        history_questions=json.load(f)

    return history_questions

def load_columns(data_path):
    schemas={}
    with open(data_path, 'r') as f:
        DBs = json.load(f)
        for db in DBs:
            schemas[db['db_id']]=[]
            table_names=db['table_names'] if 'table_names' in db else db['table_names_original']

            for column_name in db['column_names'] if 'column_names' in db else db['column_names_original'] :
                if column_name[0]==-1:
                    table_column_name=column_name[1]
                else:
                    table_column_name=table_names[column_name[0]]+' . '+column_name[1]
                schemas[db['db_id']].append(table_column_name)
            schemas[db['db_id']]=' | '.join(schemas[db['db_id']])
    return schemas
def load_data(args,data_path, schema_by_db, negative_sampling=False):
    positive_examples=[]
    negative_exmaples=[]

    dataset_by_db={}
    with open(data_path, 'r') as f:
        train_set = json.load(f)
        for item in train_set:
            if item['db_id'] not in dataset_by_db:
                dataset_by_db[item['db_id']] = []
            dataset_by_db[item['db_id']].append(item)

    for db_id in dataset_by_db:
        #full_q_for_partial_q={}
        query_for_partial_q={}
        all_full_q=set()
        all_query=set()
        for example in dataset_by_db[db_id]:
            partial_q=example['question'] #this question is a prefix
            #full_q_for_partial_q[partial_q]=set()
            query_for_partial_q[partial_q] = set()
            full_qs=example['labels']
            for full_q_q in full_qs:
                full_q=full_q_q['question']
                query=full_q_q['query']
                all_full_q.add(full_q)
                all_query.add(query)
                #full_q_for_partial_q[partial_q].add(full_q)
                query_for_partial_q[partial_q].add(query)
                positive_examples.append([' </s> '.join([partial_q,query,schema_by_db[db_id]]),1])
                #positive_examples.append([' </s> '.join([partial_q, query, schema_by_db[db_id]]), 1])

        if negative_sampling:
            for example in dataset_by_db[db_id]:
                partial_q = example['question'] #this question is a prefix

                # NOTE: here we consider the task of mapping question prefix to full SQL 
                #assert args.history_full_q_available
                for full_q in all_query:
                    if full_q not in query_for_partial_q[partial_q] and random.random() < 1.0/args.positive_over_negative/len(all_query):
                        negative_exmaples.append([' </s> '.join([partial_q,full_q,schema_by_db[db_id]]),0])
                # else:
                #     for query in all_query:
                #         if query not in query_for_partial_q[partial_q] and random.random()<0.2:
                #             negative_exmaples.append([' </s> '.join([partial_q,query,schema_by_db[db_id]]),0])


    examples=positive_examples + negative_exmaples
    print("number of examples, positive exmaples, negative examples:", len(examples),len(positive_examples),len(negative_exmaples))
    return examples

def batch_iter(data_set, batch_size,shuffle=False, seed=12345):
    index_arr = np.arange(len(data_set))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(data_set) / float(batch_size)))
    for batch_id in range(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [data_set[i] for i in batch_ids]
        batch_examples.sort(key=lambda e: -len(e[0]))

        yield batch_examples



def tensorlize(tokenizer,batch_raw,max_input_len):
    inputs=[x[0][:max_input_len] for x in batch_raw]
    labels=[x[1] for x in batch_raw]
    inputs=tokenizer(inputs,return_tensors="pt",padding=True)

    batch = [*inputs.values()]+[torch.LongTensor(labels)]
    return batch