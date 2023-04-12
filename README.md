# ACS-QG
Factorized question generation for controlled MRC training data generation.

This is the code for our paper "Asking Questions the Human Way: Scalable Question-Answer Generation from Text Corpus". Please cite this paper if it is useful for your projects. Thanks.

## How to run

1. Check and install requirements

2. Download datasets: Glove, SQuAD1.1-Zhou, SQuAD 2.0, BPE, etc. Get the wiki10000.json from https://www.dropbox.com/s/mkwfazyr9bmrqc5/wiki10000.json.zip?dl=0. I also put some data in Datasets folder. Note that the path of Datasets/ is not the same with the common/constants.py. You can revise the path to the Datasets directory.

3. Change paths in code: if your code structure is different, go to "config.py" and change "dataset_name" and other paths.
  Besides, go to "common/constants.py" and change paths.

4. About experiment platform
  I added some dirty code to handle platform problem when running experiments.
  Go to common/constants.py, set EXP_PLATFORM as "others" instead of "venus". This will helps to avoid executing any dirty code.

5. How to run
  a. debug
      Run experiments_0_debug.sh. If success, it means your environment works well.
  b. train models for once
      Run experiments_1_... at the same time. You can run them on different GPUs to save time.
  c. get sentences for once. It is used for data augmenter. Notice: change data path based on your structure.
      Run experiments_2-DA_file2sents.sh.
  d. parallel run different versions of experiments_3_repeat_da_de.sh with different arguments. This step sample inputs by sequential sampling.
      See the content in the header of experiments_3_repeat_da_de.sh.
      Search and replace the parameters to perform data augmentation and data evaluation on different parts of the generated sentences.
      We do this to help save time. We can use different GPUs to generate a lot of data in parallel.
  e. parallel generate questions using seq2seq or gpt2. You can choose just one kind of generation model.
      Run experiments_4_QG_generate_gpt2.sh to generate questions by gpt2.
      Run experiments_4_QG_generate_seq2seq.sh to generte questions by seq2seq.
  f. remove duplicated data
      Run experiments_5_uniq_seq2seq.sh to remove duplicated data.
  g. post process seq2seq results to handle the repeat problem. It is not required if you use gpt2.
      Run experiments_6_postprocess_seq2seq.sh


### To install Spcay models
- python -m spacy download en_core_web_sm
- python -m spacy download de_core_news_sm

## To donwload the specified SQuAD1.1-Zhou data
- wget https://res.qyzhou.me/redistribute.zip
- From the paper https://github.com/magic282/NQG

NOTE: You need to install Java in order to get METEOR (Eval Metric) working.

## To download ET dataset
- https://www.tensorflow.org/datasets/catalog/glue
- And thus rename the train and test txt files to tsv format and also copy the test data to act as the dev set.

# CHANGELOG
- Added to `run_glue.py`.
  ```python
  ## The config file does not contain the number of total token, hence we use the tokenizer info to initialize that info.
    if config.n_token == -1:
        config.n_token = tokenizer.vocab_size```

- data_augmentor/answer_selector.py
  in the "_post" method, in the final loop (other strategy of deleting chunk) there is a bug where the final filtering is not done using the cand_chunks list but the chunklist.
### GPT2
- The labels contained -1 for ignored indexes but in the GPT2LMHead model by Huggingface, -100 is to be used.
  Hence, replaced the -1 for lm_labels in GPT2_QG/train.py