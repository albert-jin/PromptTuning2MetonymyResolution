# Continuous Template Construction based PromptMR

import argparse
from utils_MR import get_train_test_examples, calculate_metof1_literal_acc, set_logger, log_few_shot_examples
import os
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from openprompt.plms import T5TokenizerWrapper
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
import torch
from openprompt import PromptForClassification
from transformers import AdamW
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from openprompt.data_utils.data_sampler import FewShotSampler


def experiment_prompt_base(args):
    if not args.log_file:
        logger = set_logger(f'../results_MR/prompt_ctc_init/{args.dataset}_{args.shot_count}_shot_@{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.log')
    else:
        logger = set_logger(args.log_file)
    dataset_dir = os.path.join('../dataset_MR', args.dataset)
    logger.info('--------make train/test examples...--------')
    train_examples, test_examples = get_train_test_examples(args.dataset, dataset_dir)
    if args.shot_count <= 100:  # 不大于100则是小样本学习
        sampler = FewShotSampler(num_examples_per_label=args.shot_count)
        train_examples = sampler(train_examples)
        log_few_shot_examples(train_examples, logger)
    dataset = {'train': train_examples, 'test': test_examples}
    logger.info('--------train/test examples made up!--------')
    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
    template_text = '{"placeholder":"text_a"} {"soft":"So"} {"placeholder":"text_b"} {"soft":"is"} {"soft":"a"} {"mask"} {"soft":"entity"}.'
    if args.dataset != 'ChineseMR':
        label_words = [["metonymic"], ["literal"]]
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=label_words)
    else:
        label_words = [["转喻"], ["普通"]]
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=label_words)
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    logger.info('==============================')
    logger.info(f'===== template_text: {template_text} ========')
    logger.info(f'===== label_words: {label_words} ========')
    logger.info('==============================')
    wrapped_t5tokenizer = T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,
                                             truncate_method="head")
    model_inputs = {}
    for type in ['train', 'test']:
        model_inputs[type] = []
        for sample in dataset[type]:
            tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample),
                                                                         teacher_forcing=False)
            model_inputs[type].append(tokenized_example)

    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    prompt_model = prompt_model.cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer_grouped_parameters_template = [
        {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer_temp = AdamW(optimizer_grouped_parameters_template, lr=args.lrtemp)
    logger.info("--------Set Dataloader...--------")
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                             decoder_max_length=3,
                                             batch_size=4, shuffle=False, teacher_forcing=False,
                                             predict_eos_token=False,
                                             truncate_method="head")
    logger.info("-------- Dataloader Build!--------")
    # 训练
    best_acc, best_acc_epoch = 0, 0
    performances_results = {}
    for epoch in range(args.total_epochs):
        tot_loss = 0
        prog_bar = tqdm(train_dataloader, desc=f'[Epoch. {epoch + 1}.]')
        for step, inputs in enumerate(prog_bar):
            inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_temp.step()
            optimizer_temp.zero_grad()
            prog_bar.set_postfix(ordered_dict=OrderedDict(avg_loss=tot_loss / (step + 1)))
        # 保存进度条
        logger.info(prog_bar)
        logger.info("Epoch {}, average loss: {}".format(epoch+1, tot_loss / len(train_dataloader)))
        if (epoch+1) % args.test_pre_epoch == 0:
            allpreds = []
            alllabels = []
            prog_bar_test = tqdm(test_dataloader, desc=f'[Epoch. {epoch + 1}.]')
            for inputs in prog_bar_test:
                inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            # 保存进度条
            logger.info(prog_bar_test)
            # acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
            logger.info('==============================')
            logger.info('Gold Labels:')
            logger.info(alllabels)
            logger.info('Predicted Labels:')
            logger.info(allpreds)
            logger.info('==============================')
            meto_f1, literal_f1, acc = calculate_metof1_literal_acc(alllabels, allpreds)
            performances_results[epoch+1] = [acc, meto_f1, literal_f1]
            if best_acc < acc:
                best_acc = acc
                best_acc_epoch = epoch+1
            logger.info(f'[Epoch {epoch+1}] ===> Accuracy: {acc}, Metonymic_f1: {meto_f1}, Literal_f1: {literal_f1}')
    # 测试.
    stats = "\n".join(["[Epoch {}] --> acc: {}, metoymic_f1: {}, literal_f1: {}".format(epoch, *performances_results[epoch]) for epoch in performances_results])
    logger.info(f'Performances List: \n {stats}')
    assert best_acc == performances_results[best_acc_epoch][0]
    logger.info(f'===> Best Accuracy: {best_acc}, Metonymic_f1: {performances_results[best_acc_epoch][1]}, '
                f'Literal_f1: {performances_results[best_acc_epoch][2]}, At Epoch: {best_acc_epoch}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ReLocaR', choices=['ReLocaR', 'CoNLL2003', 'ChineseMR', 'SemEval2007'], help='specify which dataset, please refer to directory dataset_MR')
    parser.add_argument('--log_file', type=str, default='', help= 'if not specify log_file, default is ./results_MR/prompt_base/{dataset_type}_{time}.log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrtemp', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=10)  # 常规 10 小样本 20    因为小样本 10~20 epoch 左右就拟合了
    parser.add_argument('--shot_count', type=int, default=999, help='if shot_count > 100, means using all dataset.') # 常规 999 小样本 <=100
    parser.add_argument('--test_pre_epoch', type=int, default=1)  # 常规 1 小样本 2
    _args = parser.parse_args()
    experiment_prompt_base(_args)
    # 具体批测试脚本见run_prompt_base.py
