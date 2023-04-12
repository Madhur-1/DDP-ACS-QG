import os

import ahocorasick
import benepar
import gensim
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


benepar.download("benepar_en3")
import sentencepiece as spm
import spacy
import torch
from spacy.symbols import ORTH

print("Start loading constants ...")

# data path
# current_path = os.getcwd()
current_path = "/storage/home/madhurjindal/ACS-QG/"
scratch_path = "/scratch/scratch8/madhurjindal/ACS-QG-Scratch/"

DATA_PATH = os.path.join(scratch_path, "Datasets")
PROJECT_PATH = current_path

CODE_PATH = os.path.join(PROJECT_PATH, "model/")
OUTPUT_PATH = os.path.join(scratch_path, "output/")
CHECKPOINT_PATH = os.path.join(scratch_path, "output/checkpoint/")
FIGURE_PATH = os.path.join(scratch_path, "output/figure/")
LOG_PATH = os.path.join(scratch_path, "output/log/")
PKL_PATH = os.path.join(scratch_path, "output/pkl/")
RESULT_PATH = os.path.join(scratch_path, "output/result/")


FUNCTION_WORDS_FILE_PATH = os.path.join(
    DATA_PATH, "original/function-words/function_words.txt"
)
FIXED_EXPRESSIONS_FILE_PATH = os.path.join(
    DATA_PATH + "/original/fixed-expressions/fixed_expressions.txt"
)

BPE_MODEL_PATH = DATA_PATH + "/original/BPE/en.wiki.bpe.vs50000.model"
BPE_EMB_PATH = DATA_PATH + "/original/BPE/en.wiki.bpe.vs50000.d100.w2v.txt"

GLOVE_NPY_PATH = DATA_PATH + "/original/Glove/glove.840B.300d"
GLOVE_TXT_PATH = DATA_PATH + "/original/Glove/glove.840B.300d.txt"

# question type
QUESTION_TYPES = [
    "Who",
    "Where",
    "When",
    "Why",
    "Which",
    "What",
    "How",
    "Boolean",
    "Other",
]
INFO_QUESTION_TYPES = ["Who", "Where", "When", "Why", "Which", "What", "How"]
BOOL_QUESTION_TYPES = [
    "Am",
    "Is",
    "Was",
    "Were",
    "Are",
    "Does",
    "Do",
    "Did",
    "Have",
    "Had",
    "Has",
    "Could",
    "Can",
    "Shall",
    "Should",
    "Will",
    "Would",
    "May",
    "Might",
]
Q_TYPE2ID_DICT = {
    "What": 0,
    "Who": 1,
    "How": 2,
    "Where": 3,
    "When": 4,
    "Why": 5,
    "Which": 6,
    "Boolean": 7,
    "Other": 8,
}

# function words
f_func_words = open(FUNCTION_WORDS_FILE_PATH, "r", encoding="utf-8")
func_words = f_func_words.readlines()

FUNCTION_WORDS_LIST = [word.rstrip() for word in func_words]


# fixed expressions
f_fixed_expression = open(FIXED_EXPRESSIONS_FILE_PATH, "r", encoding="utf-8")
fixed_expressions = f_fixed_expression.readlines()

FIXED_EXPRESSIONS_LIST = [word.rstrip() for word in fixed_expressions]


# AC Automaton
AC_AUTOMATON = ahocorasick.Automaton()
for idx, key in enumerate(FIXED_EXPRESSIONS_LIST):
    AC_AUTOMATON.add_word(key, (idx, key))
AC_AUTOMATON.make_automaton()


# BPE
SPM = spm.SentencePieceProcessor()
SPM.Load(BPE_MODEL_PATH)

# special tokens
SPECIAL_TOKENS = {"pad": "<pad>", "oov": "<oov>", "sos": "<sos>", "eos": "<eos>"}
SPECIAL_TOKEN2ID = {"<pad>": 0, "<oov>": 1, "<sos>": 2, "<eos>": 3}

# spaCy
NLP = spacy.load("en_core_web_sm")
# prevent tokenizer split special tokens
for special_token in SPECIAL_TOKENS.values():
    NLP.tokenizer.add_special_case(special_token, [{ORTH: special_token}])

# benepar
PARSER = benepar.Parser("benepar_en3")

# glove
GLOVE = gensim.models.keyedvectors.KeyedVectors.load(GLOVE_NPY_PATH)

# device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# env
EXP_PLATFORM = "others"  # set it to be "venus" or any other string. This is just used for run experiments on Venus platform.

print("Finished loading constants ...")
