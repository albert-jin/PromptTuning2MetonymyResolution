import os
from openprompt.data_utils import InputExample
import random
import numpy as np
import logging

global_uid = 0
global_log_id = 0


def preprocess(sentence):
    special_tokens = {'$', '£', '.', '-', '/', '&', '¥', '×', '¼', '’', '€', '+', '°', ',', ':', '\''}
    new_sentence = ''
    for idx, token in enumerate(sentence):
        new_sentence += token
        if idx != len(sentence) - 1:
            if (token in special_tokens and sentence[idx + 1] != ' ') or (
                    sentence[idx + 1] in special_tokens and token != ' '):
                new_sentence += ' '
    return new_sentence


def read_InputExamples(filepath):
    # 针对三个英文数据集的格式
    global global_uid
    input_examples = []
    label = 0 if 'literal' in filepath else 1
    with open(filepath, mode='rt', encoding='utf-8') as inp:
        for line in inp:
            line = line.strip()
            if line:
                entity = line.split('<SEP>')[0]
                sentence = ' '.join(line.split('<SEP>')[1].split('<ENT>'))
                new_sentence = preprocess(sentence)
                input_examples.append(InputExample(text_a=new_sentence, text_b=entity, label= label, guid=global_uid))
                global_uid += 1
    return input_examples


def read_InputExamples_CMU(filepath):
    # 只针对CMU数据集的格式
    global global_uid
    input_examples = []
    with open(filepath, mode='rt', encoding='utf-8') as inp:
        for line in inp:
            line = line.strip()
            if line:
                try:
                    [sentence, label, entity] = line.split('\t')[:3]
                    assert label in ['0', '1']
                    label = int(label)
                except Exception as e:
                    print('an error:', line,  e.args)
                    continue
                input_examples.append(InputExample(text_a=sentence, text_b=entity, label=label, guid=global_uid))
                global_uid += 1
    return input_examples


def get_train_test_examples(dataset, dataset_dir):
    assert os.path.isdir(dataset_dir)
    if dataset != 'ChineseMR':
        file_map = {'CoNLL2003':'conll', 'ReLocaR':'relocar', 'SemEval2007':'semeval'}
        literal_train_prefix, literal_test_prefix, meton_train_prefix, meton_test_prefix \
            = '{}_literal_train.txt','{}_literal_test.txt','{}_metonymic_train.txt','{}_metonymic_test.txt'
        train_examples = read_InputExamples(os.path.join(dataset_dir,literal_train_prefix.format(file_map[dataset])))\
        + read_InputExamples(os.path.join(dataset_dir,meton_train_prefix.format(file_map[dataset])))

        test_examples = read_InputExamples(os.path.join(dataset_dir,literal_test_prefix.format(file_map[dataset])))\
        + read_InputExamples(os.path.join(dataset_dir,meton_test_prefix.format(file_map[dataset])))

    else:
        train_examples = read_InputExamples_CMU(os.path.join(dataset_dir,'ChineseMR_train.txt'))
        test_examples = read_InputExamples_CMU(os.path.join(dataset_dir,'ChineseMR_test.txt'))
    random.shuffle(train_examples)
    random.shuffle(test_examples)
    return train_examples, test_examples


def calculate_metof1_literal_acc(y_true, y_pred):
    # return metonymic f1 & literal f1
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = np.sum(np.equal(y_true, y_pred)) / y_true.shape[0]
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))  # pre and gold is 1
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  # gold is 0; pre is 1
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  # gold is 1; pre is 0
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  # pre and gold is 0
    metonymic_prec, metonymic_rec = TP/(TP+FP), TP/(TP+FN)
    if metonymic_prec + metonymic_rec == 0:
        metonymic_f1 = 0
    else:
        metonymic_f1 = 2*metonymic_prec*metonymic_rec/(metonymic_rec + metonymic_prec)
    literal_prec, literal_rec = TN/(TN+FN), TN/(TN+FP)
    if literal_prec + literal_rec == 0:
        literal_f1 = 0
    else:
        literal_f1 = 2*literal_prec*literal_rec/(literal_rec + literal_prec)
    return metonymic_f1, literal_f1, acc


def calculate_metof1_literal_acc_precisionandrecall(y_true, y_pred):
    # return metonymic f1 & literal f1
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = np.sum(np.equal(y_true, y_pred)) / y_true.shape[0]
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))  # pre and gold is 1
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  # gold is 0; pre is 1
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  # gold is 1; pre is 0
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  # pre and gold is 0
    metonymic_prec, metonymic_rec = TP/(TP+FP), TP/(TP+FN)
    if metonymic_prec + metonymic_rec == 0:
        metonymic_f1 = 0
    else:
        metonymic_f1 = 2*metonymic_prec*metonymic_rec/(metonymic_rec + metonymic_prec)
    literal_prec, literal_rec = TN/(TN+FN), TN/(TN+FP)
    if literal_prec + literal_rec == 0:
        literal_f1 = 0
    else:
        literal_f1 = 2*literal_prec*literal_rec/(literal_rec + literal_prec)
    return metonymic_f1, literal_f1, acc, metonymic_prec, metonymic_rec, literal_prec, literal_rec


def set_logger(log_path):
    global global_log_id
    logger_id = str(global_log_id)
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)
    global_log_id += 1
    return logger


def log_few_shot_examples(train_examples, logger):
    logger.info('+++++++++++ A View of Few-Shot Examples ++++++++++++++')
    logger.info('==== Metonymic: ====>')
    for example in train_examples:
        if example.label == 1:
            logger.info(f'{example.text_b} <---> {example.text_a}')
    logger.info('==== Literal: ====>')
    for example in train_examples:
        if example.label == 0:
            logger.info(f'{example.text_b} <---> {example.text_a}')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++')


# if __name__ == '__main__':
#     calculate_metof1_literal_acc([1,0,1,0], [0,1,1,1])
