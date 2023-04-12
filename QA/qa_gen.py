from argparse import ArgumentParser
from collections import Counter
from pprint import pformat

import pandas as pd

from transformers import pipeline


def get_tokens(question_answerer, text):
        return question_answerer.tokenizer(text)['input_ids']

def f1_score(question_answerer, pred_text, targ_text):
        pred_tokens = get_tokens(question_answerer, pred_text)
        targ_tokens = get_tokens(question_answerer, targ_text)
        common = Counter(pred_tokens) & Counter(targ_tokens)
        num_com = sum(common.values())
        if len(pred_tokens) == 0 or len(targ_tokens)==0:
            # If either is no-answer, then f1 is 1 if they agree, 0 otherwise
            return int(pred_tokens==targ_tokens)

        if num_com==0:
            return 0
        precision = 1.0 * num_com/len(pred_tokens)
        recall = 1.0 * num_com/len(targ_tokens)
        f1 = (2 * precision * recall)/ (precision + recall)
        return f1

def run():
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="gpt2", help="gpt2 or seq2seq")
    parser.add_argument("--model_name_or_path", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--input_file", type=str, default="", help="File to use for QA filtering containing QG results")
    parser.add_argument("--output_file", type=str, default="", help="File to save with the QA filtering results appended to original file")
    args = parser.parse_args()

    print(pformat(args))

    question_answerer = pipeline("question-answering", model=args.model_name_or_path)

    
    print("Reading dataset from ", args.input_file)
    data = pd.read_csv(args.input_file, sep="\t", index_col=0)

    gen_ans= question_answerer(question=data.question.to_list(), context=data.paragraph.to_list())
    gen_ans = pd.DataFrame(gen_ans)

    data['qa_ans'] = gen_ans['answer']
    data['qa_ans_score'] = gen_ans['score']



    
    f1_scores = list(map(lambda x: f1_score(question_answerer, *x), 
                        zip(data.answer.to_list(), data.qa_ans.to_list())))

    data['qa_f1_score'] = f1_scores

    # Filtering for f1_score >= 0.9
    data[data.qa_f1_score >=0.9].to_csv(args.output_file, sep="\t")
    print("Saved to ", args.output_file)

if __name__ == "__main__":
    run()