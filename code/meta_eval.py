import numpy as np
np.random.seed(42)
import pandas as pd
import meta_evaluation_lib
from metric import correlation, acc, expected_calibration_error, brier_score
import json

def meta_eval(human_score, metric_score, data_name):
    corr = correlation(human_score=human_score, metric_score=metric_score)
    accuracy = acc(human_score=human_score, metric_score=metric_score)
    ece = expected_calibration_error(human_score=human_score, metric_score=metric_score, savepath="./%s/figure/%s.pdf"%(data_name, data_name))
    tmpdata = pd.DataFrame.from_dict({"label": human_score, data_name: metric_score})
    roc_auc = pd.DataFrame.to_dict(meta_evaluation_lib.evaluate_metrics(tmpdata, scores_columns=[data_name], thresholds=[0.5]))
    bs = brier_score(human_score=human_score, metric_score=metric_score)
    print("correlation:", corr)
    print("accuracy:", accuracy)
    print("expected calibration error:", ece)
    print("ROC AUC:", roc_auc)
    print("brier score:", bs)

def meta_pro_direct(choice):
    for k, (tok, tok_logprob, top_log_prob) in enumerate(zip(choice["logprobs"]["tokens"], choice["logprobs"]["token_logprobs"], choice["logprobs"]["top_logprobs"])):
        min_prob = np.exp(min([top_log_prob[w] for w in top_log_prob]))
        if tok in yes_tok_list + no_tok_list:
            yes_prob = np.sum([np.exp(top_log_prob[cand_tok]) if cand_tok in top_log_prob else 0. for cand_tok in yes_tok_list])
            no_prob = np.sum([np.exp(top_log_prob[cand_tok]) if cand_tok in top_log_prob else 0. for cand_tok in no_tok_list])
            if yes_prob == 0:
                yes_prob = min_prob
            if no_prob == 0:
                no_prob = min_prob
            return yes_prob, no_prob
    return None, None

def pro_eval_score(data_name):
    with open(data_name) as fin:
        human_score, metric_score = [], []
        for k, line in enumerate(fin):
            tmpdata = json.loads(line)
            label = tmpdata["label"]
            eval_response = tmpdata["eval_response"]
            all_yes_no_prob = []
            for choice in eval_response["choices"][:20]:
                yes_prob, no_prob = meta_pro_direct(choice=choice)
                all_yes_no_prob.append({"yes": yes_prob, "no": no_prob})

            yes_prob_ = np.sum(np.array([prob["yes"] for prob in all_yes_no_prob]))
            no_prob_ = np.sum(np.array([prob["no"] for prob in all_yes_no_prob]))
            yes_prob = yes_prob_ / (yes_prob_ + no_prob_)
            no_prob = no_prob_ / (yes_prob_ + no_prob_)

            human_score.append(1. if label in ["SUPPORT", "SUPPORTS", 1] else 0.)
            metric_score.append(yes_prob)

        meta_eval(human_score=human_score, metric_score=metric_score, data_name=data_name)



if __name__ == '__main__':
    yes_tok_list = ["%s%s"%(a,b) for a in [" ", "▁", ""] for b in ["Yes", "yes", "YES", "A"]]
    no_tok_list = ["%s%s"%(a,b) for a in [" ", "▁", ""] for b in ["No", "no", "NO", "B"]]

    pro_eval_score(data_name="./result/result.json", without_label=False)
