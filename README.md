# LLMforFV

This repo provides the data and codes for our NAACL 2024 work [Language Models Hallucinate, but May Excel at Fact Verification](https://arxiv.org/abs/2310.14564). 



### Data

You can find the manually annotated data under the `./data` directory.

- `id`: the same `id` corresponds to the same input to different generation models

- `model`: the generation model (one of `flant5_xxl`, `llama30b`, `llama65b`, `gpt3`)

- `statement`: the generated statement to be judged

- `final_label`: the human-annotated label of `statement` (one of `Unfactual`, `Factual`, `Not Sure`)

- `context`: the leading context of `statement` for the ParaGen (Para) data.

- `statement_id`: the position id (from 1 to 5) of `statement` for the ParaGen (Para) data. 



### Code

Here we provide an example for using FLAN-T5 to evaluate the factuality of given statements.

1. Using `eval_flant5.py` to generate verbal judgments for given statements.
2. Using `meta_eval.py` to evaluate the judgments of FLAN-T5



### Citation

Please kindly cite our paper if this paper and it is helpful.

```
@article{guan2023language,
  title={Language Models Hallucinate, but May Excel at Fact Verification},
  author={Guan, Jian and Dodge, Jesse and Wadden, David and Huang, Minlie and Peng, Hao},
  journal={arXiv preprint arXiv:2310.14564},
  year={2023}
}
```