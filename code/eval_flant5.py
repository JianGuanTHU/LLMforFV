import numpy as np
np.random.seed(42)
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


def completion_with_backoff(model, prompt, temperature, top_p, max_tokens, n, logprobs, tokenizer, min_tokens=0):
    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            do_sample=False if temperature==0 else True,
            num_beams=1,
            temperature=temperature,
            top_k=len(tokenizer),
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            num_return_sequences=1 if temperature==0 else n,
        )
    final_layer_hidden_states = torch.stack([state[-1][:, 0, :] for state in outputs["decoder_hidden_states"]], 1)
    logits = model.lm_head(final_layer_hidden_states)
    all_log_probs = torch.log_softmax(logits, dim=-1)
    vocab_size = all_log_probs.size()[-1]

    results = {"choices":[]}
    for nn in range(n):
        log_probs = all_log_probs[nn]
        # [len]
        output_sequence = outputs["sequences"][nn, 1:]
        # 32128
        # [len, vocab_size]
        one_hot_next_token = torch.nn.functional.one_hot(output_sequence, vocab_size)
        # [len]
        all_token_logprobs = torch.sum(one_hot_next_token * log_probs, -1).tolist()
        # [len, logprobs]
        value, index = torch.topk(log_probs, logprobs)
        value = value.tolist()
        index = index.tolist()
        all_top_logprobs = [{"%s"%(tokenizer.convert_ids_to_tokens([ind])[0]): val for val, ind in zip(top_val, top_ind)} for top_val, top_ind in zip(value, index)]

        text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=False, skip_special_tokens=False)
        results["choices"].append({
            "input": "",#input_text,
            "text": text[:text.find("</s>")],
            "output": "",
            "logprobs":{
                "tokens": tokenizer.convert_ids_to_tokens(output_sequence),
                "token_logprobs": all_token_logprobs,
                "top_logprobs": all_top_logprobs,
            }
        })
    return results


def trunct(ipt, maxlen):
    token = gpt2_tokenizer.tokenize(ipt.strip())
    if len(token) > maxlen:
        return gpt2_tokenizer.convert_tokens_to_string(token[-maxlen:])
    else:
        return ipt.strip()


def evaluate(data_dir, data_name):
    with open("./%s/%s.json"%(data_dir, data_name)) as fin:
        dev_data = [json.loads(line) for line in fin]

        for k, d in enumerate(dev_data):
            if "claim" in d:
                d["generate"] = {"choices":[{"text": trunct(d["claim"], 200)}]}
            else:
                d["generate"] = {"choices":[{"text": trunct(d["statement"], 200)}]}

    out_name = "%s/result.json"%(result_dir)
    with open(out_name, "w") as fout:
        for k, tmpclp in enumerate(dev_data):
            response = {"choices": []}
            retrieve_num = 10
            prompt = "Based on the given facts, is the statement correct? (A) Yes. (B) No. Please answer A or B:"

            claim_prompt = '''Facts:\n%s\nStatement: %s\n%s'''%(
                "\n".join(["%d. %s"%(i, dev_data[k]["ctxs"][i]["text"]) for i in range(min([retrieve_num, len(dev_data[k]["ctxs"])]))]), 
                dev_data[k]["generate"]["choices"][0]["text"].strip(),
                prompt)

            claim_prompt = trunct(claim_prompt, 4000)
            response["choices"] += completion_with_backoff(
                model=flant5_model,
                prompt=claim_prompt,
                temperature=0,
                top_p=1,
                max_tokens=100,
                n=1,
                logprobs=5,
                tokenizer=tokenizer,
            )["choices"]
            tmpclp["eval_response"] = response
            fout.write(json.dumps(tmpclp)+"\n")


if __name__ == '__main__':
    model_name_path = "model_name_path"
    device = "cuda:0"
    data_dir = "./data/hcs/hcs-closed/"
    result_dir = "./result"

    flant5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("./gpt2")

    data_name = "scifact_test"
    evaluate(data_dir=data_dir, data_name=data_name)
