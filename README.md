# PromptTuning2MetonymyResolution

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

# Thank you!
```
:sunglasses: :pray: :innocent: :heartpulse: :heartpulse: :heartpulse: 	:stuck_out_tongue_closed_eyes:
```