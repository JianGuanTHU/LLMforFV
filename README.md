# LLMforFV

This repo provides the data and codes (coming soon) for our work [Language Models Hallucinate, but May Excel at Fact Verification](https://arxiv.org/abs/2310.14564). 

You can find the manually annotated data under the `./data` directory.

- `id`: the same `id` corresponds to the same input to different generation models

- `model`: the generation model (one of `flant5_xxl`, `llama30b`, `llama65b`, `gpt3`)

- `statement`: the generated statement to be judged

- `final_label`: the human-annotated label of `statement` (one of `Unfactual`, `Factual`, `Not Sure`)

- `context`: the leading context of `statement` for the ParaGen (Para) data.

- `statement_id`: the position id (from 1 to 5) of `statement` for the ParaGen (Para) data. 

I am very interested in the exploration of evaluating and alleviating the hallucination problem of LLMs. Welcome to contact me if you want collaboration or communication!



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