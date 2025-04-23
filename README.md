<p align="center">
<img src="./doc/png/logo-transparent.png" width=10% />
</p>
<h1 align="center">
Generative Fuzzy System for Sequence-to-Sequence Learning
</h1>
<p align="center">
<img src="https://img.shields.io/badge/OS-Ubuntu22.4-blue" />
<img src="https://img.shields.io/badge/Python-3.8-red" />
<img src="https://img.shields.io/badge/Build-Success-green" />
<img src="https://img.shields.io/badge/License-BSD-blue" />
<img src="https://img.shields.io/badge/Release-0.1-blue" />
</p>




We proposed a novel framework (**Generative Fuzzy System**, **GenFS**) combining fuzzy systems and generative models, with the target of processing multimodal data generation.

<p align="center">
<img src="./doc/png/genfs.png" width=30% />
</p>

- [x] Fuzzys2s for text sequence **in this paper**;
- [ ] FuzzyDiffusion for image modal;
- [ ] FuzzyMusic for audio modal;
- [ ] FuzzyAgents for multimodal (text, image, audio);



## GenFS-based FuzzyS2S
FuzzyS2S is an end-to-end GenFS-based model for sequence-to-sequence learning.
<p align="center">
<img src="./doc/png/fuzzys2s.png" width=40%/>
</p>
The Figure is the structure of FuzzyS2S, $TF^k$ is the Transformer processing unit of the kth rule consequent, $s_x$ is the input sequence, and $s_y$ is the target sequence. the Preprocess module named Sequence to Vector is to implement the conversion from sequences to word vectors, and the Postprocess module named Embedding to Sequence is to convert the decoded word embeddings into the target sequences.

[Fuzzy Tokenizer] we propose a specific multi-scale tokenizer, called fuzzy tokenizer. The fuzzy tokenizer is a fuzzy system based on multi-scale sub-word tokenizers that enables the adaptive slicing of words at different scales.

## Datasets

The datasets include three categories: machine translation, summary generation, and code generation, totaling 12 datasets. The datasets for machine translation are WMT14, Tatoeba, EUconst and Ubuntu. The datasets for summary generation are CNN/DM (CNN Daily Mail), SAMSum, XLSum and BillSum. The datasets for code generation are HS (HearthStone), MTG (Magic the Game), GEO (Geoquery) and Spider. 

### (1) Machine Translation Datasets

**[WMT14](https://huggingface.co/datasets/wmt/wmt14)**: This dataset is the most commonly used dataset internationally in the field of neural machine translation. It is a collection of datasets used for the shared tasks of the 9th 2014 Workshop on Statistical Machine Translation (WMT). The dataset includes English-German, German-English, English-French, French-English, English-Hindi, Hindi-English, English-Czech, Czech, English-English, English-Russian, Russian, and Russian-English. English-Czech, Czech-English, English-Russian, and Russian-English language pairs. In this paper, the English-French language pair is chosen to evaluate the model’s capability. 

**[Tatoeba](https://huggingface.co/Helsinki-NLP/opus-tatoeba-en-tr)**: This dataset originates from the Tatoeba project, which is a dataset of multilingual translations, containing 397 languages and 4,344 bilingual texts. The Tatoeba project was founded by Trang Ho in 2006 to maintain a massive database of sentences and translations. In this paper, the English-French language pairs (2023) were selected to evaluate the modelling capabilities. 

**[EUconst](https://huggingface.co/datasets/Helsinki-NLP/euconst)**: this dataset is collected from the parallel corpus of the European Constitution (EC) and contains 210 bilingual texts in 21 languages. 

**[Ubuntu](https://huggingface.co/datasets/Helsinki-NLP/opus_ubuntu)**: This dataset consists of translations of system package information donated by the Ubuntu community in 244 languages and contains 23,988 pairs of bilingual corpus texts. In this paper, a subset of Bosnian-Assamese (bs-as) language pairs is used. Since the selected languages are relatively niche, they can be used to evaluate the model's ability to learn particular grammatical sequences. 

### (2) Summary Generation Datasets

**[CNN/DM](https://huggingface.co/datasets/abisee/cnn_dailymail)** (CNN Daily Mail): This dataset is an English corpus dataset containing more than 300,000 unique news articles written by CNN and Daily Mail journalists. The current version supports both extracted and abstracted summaries. The original version was created for machine reading and comprehension and abstracted Q&A. This dataset is a very important benchmarking dataset in the field of summary generation. 

**[SAMSum](https://huggingface.co/datasets/Samsung/samsum)**: This dataset contains approximately 16,000 sets of dialogues with summaries created and recorded by linguists fluent in English. In this dataset, the style and register of the dialogues are diverse, containing informal, semi-formal or formal dialogues. In addition, the dialogues contain slang, emoticons and spelling mistakes. 

**[XLSum](https://huggingface.co/datasets/GEM/xlsum)**: This highly multilingual aggregated dataset supports 44 languages, with data derived from BBC news reports. The Franch subset is used in this paper. 

**[BillSum](https://huggingface.co/datasets/FiscalNote/billsum)**: This dataset was collected from US Congressional and California Bill Summaries and specifically consists of three parts: the US Training Bill, the US Testing Bill, and the California Testing Bill. It focuses on medium-length legislation between 5,000 and 20,000 characters in length. 

### (3) Code Generation Datasets

**[HS](https://huggingface.co/datasets/dvitel/hearthstone)** (HearthStone) : This dataset is a code generating dataset collected from a card game simulator, with inputs consisting of a mixture of natural language and partially structured data, and outputs in python code. The dataset serves as a benchmark test generated from current research code. 

**MTG** (Magic the Game): This dataset is a code generation dataset collected from a card game engine, where the input is a mixture of natural language and partially structured data, and the output is java code. It is also a benchmarking dataset for studying code generation.

**[GEO](https://huggingface.co/datasets/DARELab/geoquery)** (Geoquery): This dataset is a standard semantic parsing benchmark test dataset containing 704 samples, which generates target code in the logical language SQL, thus evaluating the model's logical code generation capabilities.

**[Spider](https://huggingface.co/datasets/xlangai/spider)**: This dataset is a large, complex cross-domain semantic parsing text-to-SQL dataset annotated by 11 Yale students. It is an important benchmarking dataset for evaluating semantic parsing capabilities, containing 10,181 questions and 5,693 unique complex SQL queries across 200 databases. 
## Denpency
```

```

## Training

## Test

## Acknownledge

