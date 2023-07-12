# PromptTuning2MetonymyResolution
![version](https://img.shields.io/badge/version-v1.0.1-blue)
> PromptMR: Adapting Prompt Learning to Metonymy Resolution :smiley:

```
How many surprises it can bring us when adapting the Prompt Learning paradigm to the Metonymy Resolution task?
```
## Contributions

:star: :star: :star: :star: :star:

> ***PromptMRs*** are the first also systematic explorations that adapt the prompt-tuning
paradigm to the NLP Metonymy Resolution research area.

```angular2html
> For Template Engineering in Prompting, apart from the attempt at Discrete Template Construction strategies (i.e. manual-crafted template),
we also explore the potentials of the Automated Continuous Template Construction strategy.

> For Answer Engineering in Prompting, apart from the attempt of Discrete Answer Search (i.e. manual-crafted mapping strategy that uses the 
answer words set to predict final labels), we further explore the Continuous Answer Automatic Search strategy.

> Extensive experiments prove that when compared with previous SOTA methods for Metonymy Resolution, the Prompt Learning methods are still
sufficiently competitive. Concretely, our Prompt Learning based Metonymy Resolution methods perform fairly well in terms of training convergence
efficiency and final classification performance over previous SOTA methods, both in data-sufficient (i.e. full-data training) and data-scarce 
(i.e. few-shot) scenarios.

```

***

### Note & Installation

- Please use Python 3.8+ for OpenPrompt
- Make sure the following package with were installed correctly.

> pytorch                   1.9.0
```shell
pip install pytorch==1.9.0
```

> transformers              4.24.0
```shell
pip install transformers==4.24.0
```
> openprompt                       1.0.1 
> 
> (To play with the latest features, you can also install OpenPrompt from the source.)
```shell
pip install openprompt==1.0.1
```

### Using Git
Clone the repository from github:
```shell
git clone https://github.com/thunlp/OpenPrompt.git
cd OpenPrompt
pip install -r requirements.txt
python setup.py install
```

:star: :star: :star:
### Introduction by a Example of PromptMR-base

With the modularity and flexibility of the code organization, you can easily reproduce all the experiments from `/experiments_MR/` that recorded in `/results_MR/`.

- Step 1.
> At  first, you should check all the available datasets is stored inside in the project.

`|--dataset_MR`

`|--CoNLL2003 |--conll_literal_test.txt --conll_literal_train.txt --conll_metonymic_train.txt --conll_metonymic_test.txt`

`|--ReLocaR |--relocar_literal_test.txt --relocar_literal_train.txt --relocar_metonymic_test.txt --relocar_metonymic_train.txt`

`|--SemEval2007 |--semeval_literal_test.txt --semeval_literal_train.txt --semeval_metonymic_test.txt --semeval_metonymic_train.txt`

- Step 2.
> Then, navigate to the Python execution file through the specified path. 
> 
> To specify the hyperparameters for the following experiment, locate the following code. (such as Define a Pre-trained Language Models (PLMs) as backbone.)

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ReLocaR', choices=['ReLocaR', 'CoNLL2003', 'ChineseMR', 'SemEval2007'], help='specify which dataset, please refer to directory dataset_MR')
    parser.add_argument('--log_file', type=str, default='', help= 'if not specify log_file, default is ./results_MR/prompt_base/{dataset_type}_{time}.log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=10)  # 常规 10 小样本 20    因为小样本 10~20 epoch 左右就拟合了
    parser.add_argument('--shot_count', type=int, default=999, help='if shot_count > 100, means using all dataset.') # 常规 999 小样本 <=100
    parser.add_argument('--test_pre_epoch', type=int, default=1)  # 常规 1 小样本 2
    _args = parser.parse_args()
    experiment_prompt_base(_args)
```
> We have debugged it to the best of our ability, but you can still conduct various exploratory debugging if needed.

- Step 3. 
> Run the script.   
> `(This will require some time for model training and optimization, depending on your GPU hardware specifications and the size of the dataset.)`

- Step 4.
> Observe the console logs and locate the training log file at the address specified by the following code.

```python
if not args.log_file:
    logger = set_logger(f'../results_MR/prompt_base/{args.dataset}_{args.shot_count}_shot_@{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.log')
else:
    logger = set_logger(args.log_file)
```

***
## Performance

> The left shows the routine architecture of Prompt Learning, and the right represents
the specific implementation of our first exploration, PromptMR-base, based on a Metonymy
Resolution example of the sentence “Belgium beat Turkey 2 - 1.”.

[![](/assets/finprocomparisons.PNG "Fine-tuning & Prompt-tuning")][Fine-tuning & Prompt-tuning]

***

### PromptMR-base

> Fully-trained experimental comparisons of our PromptMR-base compared with
other SOTA methods averaged over 5 runs.

[![](/assets/fulltrainPromptMRbase.PNG "Fully-trained comparisons of PromptMR-base")][Fully-trained comparisons of PromptMR-base]

***
[![](/assets/promptbase.PNG "PromptMR-base")][PromptMR-base]

> Few-shot experimental comparisons of our PromptMR-base compared with other
SOTA methods averaged over 5 runs. Acc (Std) denotes the Accuracy metric ‘Acc’ with
the standard deviation ‘Std’. F1-L and F1-M denote literal and metonymic f1-score,
respectively. All numbers are measured using %.

[![](/assets/PromptMR_base_fs.PNG "Few-shot experiments of PromptMR-base")][Few-shot experiments of PromptMR-base]

***

[![](/assets/promptcasandctc.PNG "PromptMR-CTC & PromptMR-CAS")][PromptMR-CTC & PromptMR-CAS]

> The performance statistics of all the PromptMR variants under the few-shot
learning and full-data supervised learning. All results are averaged over 5 runs to maintain
experimental authenticity.

[![](/assets/PromptMRvariants.PNG "three PromptMR variants' performance")][three PromptMR variants' performance]

***

# Thank you!

:sunglasses: :pray: :innocent: :heartpulse: :heartpulse: :heartpulse: 	:stuck_out_tongue_closed_eyes:

***

## Contributors
- https://github.com/albert-jin/
- https://gitlab.com/moshidai
- https://github.com/whimSYZ
- https://github.com/zhanghaok
- https://github.com/HongLouyemeng

We thank all the contributors to this project, more contributors are welcome!