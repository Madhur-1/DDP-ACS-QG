import torch
import torch.nn.functional as F

import transformers
from transformers import (CONFIG_NAME, WEIGHTS_NAME, AdamW, GPT2Config,
                          GPT2LMHeadModel, GPT2Tokenizer)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
SPECIAL_TOKENS = [
    "<sos>",
    "<eos>",
    "<paragraph>",
    "<clue>",
    "<answer>",
    "<style>",
    "<question>",
    "<pad>",
]
SPECIAL_TOKENS_DICT = {
    "bos_token": "<sos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "additional_special_tokens": [
        "<paragraph>",
        "<clue>",
        "<answer>",
        "<style>",
        "<question>",
    ],
}

tokenizer.add_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))

sos, eos, paragraph, clue, answer, style, question, pad = tokenizer.convert_tokens_to_ids(
    SPECIAL_TOKENS
)

print("paragraph is: ", paragraph)
print("clue is: ", clue)
print("answer is: ", answer)
print("style is: ", style)
print("question is: ", question)
print("pad is: ", pad)


model.train()
batch = (
    torch.tensor(
        [
            [
                50257,
                464,
                6403,
                318,
                262,
                1688,
                5852,
                286,
                262,
                2908,
                43068,
                286,
                7439,
                6372,
                357,
                45781,
                407,
                663,
                1743,
                10043,
                11,
                543,
                389,
                287,
                10598,
                737,
                50261,
                49,
                462,
                50260,
                1169,
                2908,
                43068,
                286,
                7439,
                6372,
                50262,
                8496,
                50263,
                8496,
                318,
                262,
                10043,
                286,
                262,
                2908,
                43068,
                286,
                262,
                7439,
                6372,
                30,
                50258,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
            [
                50257,
                2953,
                404,
                262,
                8774,
                11819,
                338,
                3869,
                29500,
                318,
                257,
                10861,
                15207,
                286,
                262,
                5283,
                5335,
                13,
                50261,
                64,
                10861,
                15207,
                286,
                262,
                5283,
                5335,
                50260,
                13383,
                50262,
                2061,
                50263,
                2061,
                10718,
                319,
                1353,
                286,
                262,
                8774,
                11819,
                379,
                23382,
                20377,
                30,
                50258,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
            [
                50257,
                19620,
                5535,
                11,
                262,
                13325,
                2615,
                319,
                7611,
                290,
                5140,
                1474,
                262,
                15191,
                286,
                520,
                13,
                5335,
                13546,
                11,
                7777,
                22952,
                36870,
                13517,
                13,
                50261,
                19620,
                5535,
                50260,
                727,
                395,
                50262,
                2061,
                50263,
                2061,
                318,
                262,
                13325,
                4645,
                379,
                23382,
                20377,
                30,
                50258,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
            [
                50257,
                464,
                5193,
                3710,
                12,
                5143,
                12527,
                2291,
                1115,
                14741,
                11,
                1111,
                257,
                5243,
                290,
                5581,
                4429,
                11,
                290,
                1811,
                16695,
                290,
                22790,
                13,
                50261,
                15542,
                50260,
                50139,
                12,
                5143,
                50262,
                2437,
                50263,
                2437,
                867,
                3710,
                1705,
                9473,
                389,
                1043,
                379,
                23382,
                20377,
                30,
                50258,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
        ],
        device="cpu",
    ),
    torch.tensor(
        [
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                8496,
                318,
                262,
                10043,
                286,
                262,
                2908,
                43068,
                286,
                262,
                7439,
                6372,
                30,
                50258,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
            ],
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                2061,
                10718,
                319,
                1353,
                286,
                262,
                8774,
                11819,
                379,
                23382,
                20377,
                30,
                50258,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
            ],
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                2061,
                318,
                262,
                13325,
                4645,
                379,
                23382,
                20377,
                30,
                50258,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
            ],
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                2437,
                867,
                3710,
                1705,
                9473,
                389,
                1043,
                379,
                23382,
                20377,
                30,
                50258,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
            ],
        ],
        device="cpu",
    ),
    torch.tensor(
        [
            [
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50260,
                50260,
                50260,
                50260,
                50260,
                50260,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50261,
                50261,
                50261,
                50261,
                50261,
                50260,
                50260,
                50260,
                50260,
                50260,
                50260,
                50260,
                50262,
                50262,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
            [
                50259,
                50259,
                50259,
                50259,
                50260,
                50259,
                50259,
                50259,
                50259,
                50259,
                50261,
                50261,
                50261,
                50261,
                50261,
                50261,
                50261,
                50259,
                50261,
                50261,
                50261,
                50261,
                50261,
                50261,
                50261,
                50261,
                50260,
                50260,
                50262,
                50262,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
            [
                50259,
                50261,
                50261,
                50259,
                50259,
                50260,
                50260,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50261,
                50261,
                50261,
                50260,
                50260,
                50260,
                50262,
                50262,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
            [
                50259,
                50259,
                50259,
                50260,
                50260,
                50260,
                50259,
                50259,
                50261,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50259,
                50261,
                50261,
                50260,
                50260,
                50260,
                50260,
                50262,
                50262,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50263,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
                50264,
            ],
        ],
        device="cpu",
    ),
)
# batch = tuple(input_tensor for input_tensor in batch)
input_ids = batch[0]
lm_labels = batch[1]
token_type_ids = batch[2]
print(lm_labels)
lm_loss = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids)
# print(lm_loss.shape)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

print(loss_fn(F.softmax(lm_loss), lm_labels))