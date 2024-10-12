## Table of Contents

- [SEPLN 2024](#sepln-2024)
- [Models](#models)
- [LLMs](#llms)
- [ASR (Automatic Speech Recognition)](#asr-automatic-speech-recognition)
- [Fine-tuning](#fine-tuning)
- [Datasets](#datasets)
- [Libraries/Frameworks](#libraries-frameworks)
- [Metrics](#metrics)
- [Highlights](#highlights)
- [Ideas for us](#ideas-for-us)

## SEPLN 2024

This document aims to describe the main ideas and concepts that I was able to take
from [SEPLN 2024](https://sepln2024.infor.uva.es/en/front-page-english/), which took place at the University of
Valladolid in September 2024. I’ll try to group them in different categories and also put some conclusions at the end of
the document based on what we could do with them.

### Models

**English/Multilingual:**

- [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large)
- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
- [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)
- [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [google/mt5-large](https://huggingface.co/google/mt5-large)
- [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)
- [LongFormer](https://huggingface.co/docs/transformers/en/model_doc/longformer): transformer for long sequences of
  text.
- [Babelscape/mrebel-large](https://huggingface.co/Babelscape/mrebel-large): relation extraction. Given a text or
  document, the model identifies entities (subject and object) and their relations (predicate).  
  *Example:* Given: “Einstein discovered the theory of relativity"  
  Output: (Einstein, discovered, theory of relativity)
- [PrivBERT](https://huggingface.co/mukund/privbert): privacy policy language model.
- [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)

**Spanish:**

- [MarIA](https://arxiv.org/abs/2107.07253)
- [BETO](https://github.com/dccuchile/beto) (Spanish BERT)
- [BERTIN](https://huggingface.co/bertin-project)
- [ALBETO](https://github.com/dccuchile/lightweight-spanish-language-models) (Spanish ALBERT)
- [DistilBETO](https://github.com/dccuchile/lightweight-spanish-language-models) (Spanish DistilBERT)

*For some of the models presented before, some authors also experimented with the **base/large** versions.*

---

### LLMs

- [bigscience/bloom](https://huggingface.co/bigscience/bloom)
- [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)
- [BAAI/JudgeLM-7B-v1.0](https://huggingface.co/BAAI/JudgeLM-7B-v1.0)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [Iker/ClickbaitFighter-10B](https://huggingface.co/Iker/ClickbaitFighter-10B)
- [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
- [cerebras/Cerebras-GPT-1.3B](https://huggingface.co/cerebras/Cerebras-GPT-1.3B)
- [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
- [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf)
- [projecte-aina/FLOR-6.3B](https://huggingface.co/projecte-aina/FLOR-6.3B)

### ASR (Automatic Speech Recognition)

- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large)

### Fine-tuning

- Lora
- [QLora](https://github.com/artidoro/qlora/blob/main/README.md): an efficient fine-tuning approach that reduces memory
  usage enough to fine-tune a 65B parameter model on a single 48GB GPU while preserving full 16-bit fine-tuning task
  performance.

### Datasets

- [DIMEMEX](https://sites.google.com/inaoe.mx/dimemex-2024/): offensive memes in México.
- NECO: for toxicity in Spanish news
- [COSER](http://www.corpusrural.es/) (ASR) | [PRESEEA](https://preseea.uah.es/) | COREC: for ASR in people living in
  rural environments.
- [PRAUTOCAL](https://uvadoc.uva.es/handle/10324/47128) | [FLEURS](https://ar5iv.labs.arxiv.org/html/2205.12446) | [VOXPOPULI](https://github.com/facebookresearch/voxpopuli):
  for ASR in people with Down Syndrome.
- [webis/tldr-17](https://huggingface.co/datasets/webis/tldr-17) | [CLEF e-risk](https://erisk.irlab.org/) (MentalRisk)
- COALA
- NÓS: galician
- [LC-QuAD](https://paperswithcode.com/dataset/lc-quad-2-0): Q&A dataset with questions and SPARQL queries.
- [PrivaSeer](https://arxiv.org/abs/2004.11131): privacy policy dataset.
- [CoLA](https://nyu-mll.github.io/CoLA/)(corpus of linguistic
  acceptability) | [CatCOLA](https://proyectoilenia.es/publicaciones/catcola-catalan-corpus-of-linguistic-acceptability/) (
  Catalan)
- [Open-WikiTable](https://github.com/sean0042/Open_WikiTable) (Q&A over SQL tables) | Spa-Databench (Spanish version)
- [GLoHBCD](https://github.com/SelinaMeyer/GLoHBCD): health behavior change dataset (German).

### Libraries/Frameworks

- [Genaios/TextMachina](https://github.com/Genaios/TextMachina): framework for creating high-quality datasets for
  Machine Generated Text (MGT)-related tasks.
- [text2graph-API](https://pypi.org/project/text2graphapi/0.1.1/): text document to co-occurrence graph.
- [fasttext-langdetect](https://pypi.org/project/fasttext-langdetect/): language detection with fasttext by Facebook.
- [pysentimiento](https://pypi.org/project/pysentimiento/0.5.2rc3/): toolkit for sentiment analysis and social NLP
  tasks.
- [pyevall](https://pypi.org/project/PyEvALL/): evaluation tool with a range of metrics.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): framework to test generative language
  models on multiple evaluation tasks.
- [vaderSentiment](https://github.com/cjhutto/vaderSentiment): sentiment analysis tool for social media.
- [mrebel](https://github.com/Babelscape/rebel): Relation Extraction as a seq2seq task.
- [Awesome-align](https://github.com/neulab/awesome-align): tool for word alignment from multilingual BERT (mBERT).

### Metrics

- [ICM](https://aclanthology.org/2022.acl-long.399/) (Information Contrast Model): designed to evaluate hierarchical
  classification systems by taking into account the structure of the
  hierarchy. <https://dl.acm.org/doi/10.1007/s10791-020-09375-z>
- [BertScore](https://huggingface.co/spaces/evaluate-metric/bertscore): more semantic than something like BLEU/ROUGE.
  Uses BERT embeddings to compare the generated text against the reference.
- [Sentence Mover’s Similarity](https://aclanthology.org/P19-1264v2.pdf): method for evaluating text similarity,
  particularly in cases where the traditional word-level or token-level metrics (like BLEU or ROUGE) may not effectively
  capture the meaning of sentences. This metric is inspired by the Word Mover's Distance (WMD) but operates at the
  sentence level, making it more suitable for evaluating tasks like document or paragraph similarity, text
  summarization, and paraphrase detection.
- [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate#:~:text=Word%20error%20rate%20\(WER\)%20is,completely%20different%20with%20no%20similarity.) (
  for NER).
- [ROUGE-1](https://en.wikipedia.org/wiki/ROUGE_\(metric\)): overlap of unigrams between system and reference.
- F1.5, which prioritizes recall over precision (just tune the beta param
  in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)).
- [GMRev](https://pypi.org/project/GMRev/): evaluation of LLM generated responses for Q\&A systems.

### Highlights

- Ensemble of multilingual transformer models was a popular architecture design.
- Give special focus to data pre-processing/cleanup and formally defined annotation processes.
- Multiple researchers compared the use of LLMs versus BERT-based encoders for multiple tasks. Conclusion is that the
  latter still hold up really well, sometimes outperforming LLMs.
- [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html) was a solid baseline, pretty much always
  used in classification problems.
- [Learning with disagreement](https://aclanthology.org/2023.semeval-1.314/) was an interesting concept used in some
  [presentations](https://www.youtube.com/live/2rE0Cp45RUM?feature=shared&t=475).
- For automatic
  translations, [DeepL](https://www.deepl.com/en/translator?ref=aix.hu\&utm_term=\&utm_campaign=ES%7CPMAX%7CC%7CEnglish\&utm_source=google\&utm_medium=paid\&hsa_acc=1083354268\&hsa_cam=21575885174\&hsa_grp=\&hsa_ad=\&hsa_src=x\&hsa_tgt=\&hsa_kw=\&hsa_mt=\&hsa_net=adwords\&hsa_ver=3\&gad_source=1\&gclid=CjwKCAjw9eO3BhBNEiwAoc0-jfRzKqIojahCNHYHIBLLpFq_Pw-l6D3b8FnE5mS4AhSY2mEwjDZHuRoCuHQQAvD_BwE)
  was preferred (yandex for east languages).
- Validate SOTA for Spanish oriented models in [ODESIA](https://portal.odesia.uned.es/) Portal.
- Conclusion in one of the presentations is that manual evaluation is still needed for assessing LLM generation.
  E.g., generation of counter narrative: <https://ceur-ws.org/Vol-3756/RefutES2024_paper1.pdf>