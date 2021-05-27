import random
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import math
import json,json_lines
from transformers import RobertaForTokenClassification,RobertaForSequenceClassification
from transformers import RobertaTokenizer
from arg import init_arg_parser
from utils import batch_iter,tensorlize,load_columns,load_data,save_history_questions,load_history_questions,basic_statistics
from process_sql import get_sql

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def train(args,schema_by_db):
    train_set = load_data(args,os.path.join(args.train_path, 'train.json'), schema_by_db, negative_sampling=True)
    dev_set = load_data(args,os.path.join(args.train_path,  'dev.json'), schema_by_db, negative_sampling=True)

    print(len(train_set), len(dev_set))
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)
    if args.cuda:
        model = RobertaForSequenceClassification.from_pretrained(args.pretrain_model, num_labels=2).cuda()
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.pretrain_model, num_labels=2)
    print("Data Processing...")
    criterion = nn.CrossEntropyLoss(reduction='mean')
    lr=args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MAX_EPOCH=args.max_epoch
    REPORT_step=args.report_step
    BATCH_SIZE=args.batch_size


    print("Start training...")

    epoch_results=[]
    best_epoch=0
    for epoch in range(MAX_EPOCH):
        acc=[]
        losses=[]
        model.train()
        step=0
        for batch_raw in batch_iter(train_set,batch_size=BATCH_SIZE,shuffle=True):
            batch=tensorlize(tokenizer,batch_raw,args.max_input_len)
            model.zero_grad()
            optimizer.zero_grad()
            # move batch to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch = [x.to(device) for x in batch]

            batch_labels = batch[2]
            classification_logits = model(batch[0], batch[1],labels=batch[2])[1]
            results = torch.argmax(classification_logits, dim=-1).cpu().numpy()

            labels = batch_labels.cpu().numpy()
            acc.extend(np.equal(results, labels).tolist())

            loss = criterion(classification_logits, batch_labels)
            losses.append(loss.item())
            if step % REPORT_step ==0:
                print("step {j}/{total} loss: {loss}, ACC: {acc}".format(j=step, total=int(len(train_set) / BATCH_SIZE), loss=sum(losses) / len(losses),
                                                                         acc=1.0 * sum(acc) / len(acc)), file=sys.stdout)
                sys.stdout.flush()
                acc = []
                losses = []
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            step+=1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr *(MAX_EPOCH-epoch)/MAX_EPOCH

        torch.save(model.state_dict(), os.path.join(args.save_to,"epoch_{epoch}.pkl".format(epoch=epoch)))
        model.zero_grad()
        optimizer.zero_grad()
        model.eval()
        acc = []
        losses = []
        for batch_raw in batch_iter(dev_set,batch_size=BATCH_SIZE,shuffle=True):
            batch=tensorlize(tokenizer,batch_raw,args.max_input_len)
            # move batch to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch = [x.to(device) for x in batch]

            batch_labels = batch[2]
            classification_logits = model(batch[0], batch[1],labels=batch[2])[1]
            results = torch.argmax(classification_logits, dim=-1).cpu().numpy()


            labels = batch_labels.cpu().numpy()
            acc.extend(np.equal(results,labels).tolist())
            loss = criterion(classification_logits, batch_labels)
            losses.append(loss.item())
        dev_loss=sum(losses)/len(losses)
        acc = 1.0 * sum(acc) / len(acc)
        print("Epoch {epoch}, lr: {lr} loss: {loss}, ACC: {acc}".format(epoch=epoch,lr=lr*(MAX_EPOCH-epoch)/MAX_EPOCH, loss=dev_loss, acc=acc),file=sys.stdout)
        sys.stdout.flush()
        if len(epoch_results)==0 or acc >= max(epoch_results):
            best_epoch=epoch
            epoch_results.append(acc)


    print("epoch {best_epoch} has the best result on dev set".format(best_epoch=best_epoch),file=sys.stdout)
    return best_epoch

def test(args,load_model_path,data_path,schema_by_db,history_questions):
    print("Testing... Load model from {}".format(load_model_path))
    with open(data_path,"r") as f:
        test_set=json.load(f)

    if args.cuda:
        model = RobertaForSequenceClassification.from_pretrained(args.pretrain_model, num_labels=2).cuda()
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.pretrain_model, num_labels=2)
    saved_state = torch.load(load_model_path)
    #f_decode=open(args.decode_path,'w')
    model.load_state_dict(saved_state)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    all_full_q_qs={}
    for db_id in history_questions:
        all_full_q_qs[db_id]=set()
        for example in history_questions[db_id]:
            full_qs=example['labels']
            for full_q_q in full_qs:
                query=full_q_q['query']
                all_full_q_qs[db_id].add(query)

    for db_id in all_full_q_qs:
        print(db_id, "contains", len(all_full_q_qs[db_id]), "query pairs")
    # print(len(all_full_q_qs["geo"]))
    with torch.no_grad():
        decodes = []
        RECALL=[]
        for _ in range(25):
            RECALL.append(0)

        for example in test_set:
            decode = {
                "db_id": example['db_id'],
                "question": example['question'],
                "labels": example["labels"],
                "decodes": []
            }
            db_id=example['db_id']
            partial_q = example['question']   #prefix
            if db_id not in all_full_q_qs:
                continue
            all_full_q_q=list(all_full_q_qs[db_id])
            correct_query=set()
            for full_q in example["labels"]:
                correct_query.add(full_q["query"])
            if len(all_full_q_q)==0:
                continue
            raw_batch = []
            for query in all_full_q_q:
                schema = schema_by_db[db_id]
                if query in correct_query:
                    # if args.history_full_q_available:
                    # NOTE: now query is the only standard
                    raw_batch.append([' </s> '.join([partial_q, query, schema]), 1])
                    # else:
                    #     raw_batch.append([' </s> '.join([partial_q, query, schema]), 1])
                else:
                    # if args.history_full_q_available:
                    raw_batch.append([' </s> '.join([partial_q, query, schema]), 0])
                    # else:
                    #     raw_batch.append([' </s> '.join([partial_q, query, schema]), 0])

            prob=[]
            for round in range(math.ceil(len(raw_batch)/args.batch_size)):
                batch = tensorlize(tokenizer, raw_batch[args.batch_size*round : args.batch_size*(round+1)], args.max_input_len)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch = [x.to(device) for x in batch]

                batch_labels = batch[2]
                classification_logits = torch.softmax(model(batch[0], batch[1], labels=batch[2])[1],dim=-1)

                prob.extend([(i+args.batch_size*round,x) for i,x in enumerate(classification_logits[:,1].tolist())])
            prob=sorted(prob,key=lambda x: x[1], reverse=True)
            #print(prob)
            ordered_ids=[x[0] for x in prob]

            for id in ordered_ids[:10]:
                decode['decodes'].append(all_full_q_q[id])
            decodes.append(decode)

            for k in [1,5,10]:
                recall=0
                for i,example in enumerate(raw_batch):
                    if example[1] == 1 and i in ordered_ids[:k]:
                        recall+=1

                RECALL[k]+=1.0*recall/len(correct_query)

        for i in range(len(RECALL)):
            RECALL[i]=1.0*RECALL[i]/len(test_set)

        print("recall 1: ",RECALL[1])
        print("recall 5: ", RECALL[5])
        print("recall 10: ", RECALL[10])
        with open(args.decode_path,'w')as f:
            json.dump(decodes,f,indent=4)

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stdout)
    basic_statistics(args)
    #SQL_template_set = get_common_template(os.path.join(args.train_path, 'train.jsonl'))

    if args.mode == 'train':
        if not os.path.exists(args.save_to):
            os.makedirs(args.save_to)
        history_questions=save_history_questions(args)
        print(len(history_questions))

        schema_by_db = load_columns(args.table_path)

        best_epoch=train(args,schema_by_db)


        load_model_path = os.path.join(args.save_to,"epoch_{best_epoch}.pkl".format(best_epoch=best_epoch))

        with open(os.path.join(args.save_to,"best_epoch.txt"),'w') as f:
            f.write("epoch_{best_epoch}.pkl".format(best_epoch=best_epoch))
        test(args,load_model_path,os.path.join(args.train_path, 'dev.json'),schema_by_db,history_questions)

    elif args.mode == 'test':
        schema_by_db = load_columns(args.table_path)
        history_questions=load_history_questions(args.history_questions)
        with open(os.path.join(args.save_to,"best_epoch.txt"),'r') as f:
            load_model_path=f.readline().strip()
        load_model_path=os.path.join(args.save_to,load_model_path)
        test(args,load_model_path,os.path.join(args.train_path, 'test.json'),schema_by_db,history_questions)
    else:
        raise RuntimeError('unknown mode')