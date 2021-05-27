# coding=utf-8
import argparse
def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    # general setting
    arg_parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Run mode')
    arg_parser.add_argument('--seed', default=12345, type=int, help='random seed')
    # model setting
    arg_parser.add_argument('--pretrain_model', type=str, default='roberta-base', help='a set of domains')
    arg_parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--class_rate', default=1, type=float, help='weight for non-zero classes')
    arg_parser.add_argument('--max_input_len', default=-1, type=int, help='max length for input')
    arg_parser.add_argument('--positive_over_negative', default=0.2, type=float, help='ratio of positive examples in training process')
    # training setting
    arg_parser.add_argument('--cuda', action='store_true', default=False)
    arg_parser.add_argument('--table_path', type=str, help='path to the table file')
    arg_parser.add_argument('--train_path', type=str, help='path to the training file')
    arg_parser.add_argument('--continue_training_model', type=str, help='path to the pre-trained file')
    arg_parser.add_argument('--dev_path', type=str, help='path to the dev file')
    arg_parser.add_argument('--max_epoch', default=20, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--report_step', default=50, type=int, help='report each [report_step] step')
    arg_parser.add_argument('--save_to',type=str, help='path to save checkpoints')
    arg_parser.add_argument('--history_questions', type=str, help='file to store all questions in training data')
    arg_parser.add_argument('--history_full_q_available', action='store_true', default=False, help='')
    # evaluation setting
    arg_parser.add_argument('--load_model_path', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--test_path', type=str, help='path to the test file')
    arg_parser.add_argument('--decode_path', type=str, help='path to write decode results')


    return arg_parser