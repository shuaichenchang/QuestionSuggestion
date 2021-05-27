import random
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import math
import json,json_lines
from transformers import RobertaForTokenClassification,RobertaForSequenceClassification,RobertaModel
from transformers import RobertaTokenizer
from arg import init_arg_parser
from  utils import batch_iter,tensorlize,load_columns,load_data,save_history_questions,load_history_questions,basic_statistics
from process_sql import get_sql

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

class TwoTower(nn.Module):
    def __init__(self,args):
        super(TwoTower, self).__init__()

        if args.cuda:
            self.question_encoder = RobertaModel.from_pretrained(args.pretrain_model).cuda()
            self.sql_encoder = RobertaModel.from_pretrained(args.pretrain_model).cuda()
        else:
            self.question_encoder = RobertaModel.from_pretrained(args.pretrain_model)
            self.sql_encoder = RobertaModel.from_pretrained(args.pretrain_model)
    def forward(self,question_input_ids,question_attention_mask,sql_input_ids,sql_attention_mask):
        question_embedding = self.question_encoder(question_input_ids,question_attention_mask).last_hidden_state[:,0,:]
        sql_embedding = self.sql_encoder(sql_input_ids,sql_attention_mask).last_hidden_state[:,0,:]

        logits=torch.bmm(question_embedding.view(question_embedding.size()[0],1,question_embedding.size()[1]),
                         sql_embedding.view(sql_embedding.size(0),sql_embedding.size(1),1 ))
        return logits.view(-1)
        #return nn.functional.sigmoid(logits.view(logits.size(0),1))
    def get_question_embedding(self,question_input_ids,question_attention_mask):
        question_embedding = self.question_encoder(question_input_ids, question_attention_mask).last_hidden_state[:, 0, :]
        return question_embedding
    def get_sql_embedding(self,sql_input_ids,sql_attention_mask):
        sql_embedding = self.sql_encoder(sql_input_ids, sql_attention_mask).last_hidden_state[:, 0, :]
        return sql_embedding

def train(args,schema_by_db):
    train_set = load_data(args,os.path.join(args.train_path, 'train.json'), schema_by_db, negative_sampling=True)
    #train_set_question=
    dev_set = load_data(args,os.path.join(args.train_path,  'dev.json'), schema_by_db, negative_sampling=True)

    print(len(train_set), len(dev_set))
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_model)

    model=TwoTower(args)

    print("Data Processing...")
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
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
        for batch_raw_question,batch_raw_sql in batch_iter(train_set,batch_size=BATCH_SIZE,shuffle=True):
            batch_question=tensorlize(tokenizer,batch_raw_question,args.max_input_len)
            batch_sql = tensorlize(tokenizer, batch_raw_sql, args.max_input_len)

            model.zero_grad()
            optimizer.zero_grad()
            # move batch to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch_question = [x.to(device) for x in batch_question]
            batch_sql = [x.to(device) for x in batch_sql]

            batch_labels = batch_question[2].type(torch.float)

            #print(batch_question[0].size(),batch_question[1].size(),batch_sql[0].size(),batch_sql[1].size())
            #print('='*20)
            classification_logits= model(batch_question[0],batch_question[1],batch_sql[0],batch_sql[1])

            #classification_logits = model(batch[0], batch[1],labels=batch[2])[1]
            results = torch.gt(classification_logits,torch.zeros_like(classification_logits)).type(torch.int).cpu().numpy()
                #torch.argmax(classification_logits, dim=-1).cpu().numpy()
            #print(results)

            labels = batch_labels.cpu().numpy()
            #print(labels)
            #print('='*20)
            acc.extend(np.equal(results, labels).tolist())
            #print(classification_logits,classification_logits)
            #print(batch_labels)
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

        for batch_raw_question,batch_raw_sql in batch_iter(dev_set,batch_size=BATCH_SIZE,shuffle=True):
            batch_question=tensorlize(tokenizer,batch_raw_question,args.max_input_len)
            batch_sql = tensorlize(tokenizer, batch_raw_sql, args.max_input_len)


            # move batch to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch_question = [x.to(device) for x in batch_question]
            batch_sql = [x.to(device) for x in batch_sql]

            batch_labels = batch_question[2].type(torch.float)

            classification_logits = model(batch_question[0], batch_question[1], batch_sql[0], batch_sql[1])

            results = torch.gt(classification_logits, torch.zeros_like(classification_logits)).type(torch.int).cpu().numpy()

            #results = torch.argmax(classification_logits, dim=-1).cpu().numpy()


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

    model=TwoTower(args)
    print(load_model_path)
    saved_state = torch.load(load_model_path)
    model.load_state_dict(saved_state)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sqls_per_db={}
    sql_embeddings=[]
    sql_to_id={}
    id_to_sql=[]
    for db_id in history_questions:
        sqls_per_db[db_id]=set()
        for example in history_questions[db_id]:
            full_qs=example['labels']
            for full_q_q in full_qs:
                query=full_q_q['query']
                if query not in sql_to_id:
                    sqls_per_db[db_id].add(query)

                    #inputs=tensorlize(tokenizer, [[query,0]], args.max_input_len)

                    sql_to_id[query] = len(id_to_sql)
                    id_to_sql.append(query)

                    #sql_embedding=model.get_sql_embedding(inputs[0].to(device),inputs[1].to(device))
                    #sql_embeddings.append(sql_embedding)

    for db_id in sqls_per_db:
        print(db_id, "contains", len(sqls_per_db[db_id]), "query pairs")
    # print(len(all_full_q_qs["geo"]))
    with torch.no_grad():
        for query in id_to_sql:
            inputs = tensorlize(tokenizer, [[query, 0]], args.max_input_len)
            sql_embedding = model.get_sql_embedding(inputs[0].to(device), inputs[1].to(device))
            sql_embeddings.append(sql_embedding)

    sql_embeddings = torch.cat(sql_embeddings)
    print(len(sql_embeddings))
    print(sql_embeddings.size())
    #print(sqls_per_db)

    with torch.no_grad():
        decodes = []
        RECALL=[]
        for _ in range(25):
            RECALL.append(0)

        for example in test_set:
            decode={
                "db_id" : example['db_id'],
                "question": example['question'],
                "labels":example["labels"],
                "decodes":[]
            }
            db_id=example['db_id']
            partial_q = example['question']   #prefix
            if db_id not in sqls_per_db:
                continue
            sqls=list(sqls_per_db[db_id])
            correct_query_ids=set()
            for full_q in example["labels"]:
                if full_q["query"] in sql_to_id:
                    correct_query_ids.add(sql_to_id[full_q["query"]])
                else:
                    correct_query_ids.add(-1)
            #correct_query_ids=list(correct_query_ids)
            if len(sqls)==0:
                continue

            #get question embedding
            inputs = tensorlize(tokenizer, [[partial_q, 0]], args.max_input_len)

            question_embedding = model.get_question_embedding(inputs[0].to(device), inputs[1].to(device))
            question_embeddings = question_embedding.repeat(len(sql_embeddings),1)
            #print(question_embeddings.size(),sql_embeddings.size())
            logits = torch.bmm(question_embeddings.view(question_embeddings.size()[0], 1, question_embeddings.size()[1]),
                      sql_embeddings.view(sql_embeddings.size(0), sql_embeddings.size(1), 1))
            #print(logits.size())
            logits=[(i,x) for i,x in enumerate(logits.tolist())]
            logits = sorted(logits, key=lambda x: x[1], reverse=True)
            ordered_ids = [x[0] for x in logits]
            #print(ordered_ids[:10])
            for id in ordered_ids[:10]:
                decode['decodes'].append(id_to_sql[id])
            decodes.append(decode)
            #print(ordered_ids)
            #print(correct_query_ids)
            for k in [1, 5, 10]:
                recall = 0
                for query_id in correct_query_ids:
                    if query_id in ordered_ids[:k]:
                        recall += 1

                RECALL[k] += 1.0 * recall / len(correct_query_ids)


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