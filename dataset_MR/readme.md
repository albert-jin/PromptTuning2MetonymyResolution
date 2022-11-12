# Dataset Intros ❤️
> all the datasets listed below are available from Github:
> https://github.com/milangritta/Minimalist-Location-Metonymy-Resolution/tree/master/data
## 1. CoNLL2003

>CoNLL2003 was released together with RELOCAR (Gritta et al., 2017) and also focused on locations. It
contains about 7000 sentences taken from the CoNLL 2003 Shared Task on NER and was annotated by
one annotator only, with no quantification of the quality of the labels, and is thus potentially noisy.

### Citation

```
@inproceedings{vancouver2017relocar,
    title = "{V}ancouver Welcomes You! Minimalist Location Metonymy Resolution",
    author = "Gritta, Milan  and
      Pilehvar, Mohammad Taher  and
      Limsopatham, Nut  and
      Collier, Nigel",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1115",
    doi = "10.18653/v1/P17-1115",
    pages = "1248--1259",
}
```

## 2. ReLocaR
>ReLocaR (Gritta et al., 2017) is a Wikipedia-based dataset. Compared with SEMEVALLOC, it
is intended to have better label balance, and annotation quality, without the fine-grained analysis of
metonymic patterns. It contains 2026 sentences, and is focused on locations only. It is important to
note that the class definitions for RELOCAR are a bit different from those for SEMEVALLOC. The main
difference is in the interpretation of political entity (e.g. Moscow opposed the sanctions), which
is considered to be a literal reading in SEMEVAL, but metonymic in RELOCAR. The argument is that
governments/nations/political entities (in the case of our example, “the government of Russia”) are much
closer to organizations or people semantically, and thus metonymic.

### Citation

```
the same as the above citation.
```

## 3. SemEval2007
>SemEval2007 was first introduced by Nissim and Markert (2003) and subsequently used in SemEval 2007
Task 8 (Markert and Nissim, 2007). It contains about 3800 sentences from the BNC across two types of
entities: organizations and locations. In addition to coarse-level labels of metonym or literal, it contains
finer-grained labels of metonymic patterns, such as place-for-people, place-for-event, or place-for-product.
This is the only dataset where have such fine-grained labels of metonymy.

### Citation

```
@inproceedings{markertSemeval2007,
    title = "{S}em{E}val-2007 Task 08: Metonymy Resolution at {S}em{E}val-2007",
    author = "Markert, Katja  and
      Nissim, Malvina",
    booktitle = "Proceedings of the Fourth International Workshop on Semantic Evaluations ({S}em{E}val-2007)",
    month = jun,
    year = "2007",
    address = "Prague, Czech Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S07-1007",
    pages = "36--41",
}
```

## 4. ChineseMR
>ChineseMR (Chinese Language) is derived from the **ReLocaR** dataset. 
> 
> More details can be found in *./ChineseMR/readme.md*
