<h1 align="center">Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering</h1>

Our [paper](https://aclanthology.org/2025.acl-short.50/) shows how reasoning models can use additional test-time compute to improve their confidence allocation and deliver stronger performance in selective question answering.

![alt text](https://github.com/wjurayj/final_answer/blob/main/images/utility_surface.png)


## Installation

Be sure to use our version of vllm, altered from the original [s1 repo](https://github.com/simplescaling/s1)

```bash
pip install -r requirements.txt
cd eval/lm-evaluation-harness
pip install -e .[vllm]
```

## Usage
First run:

```bash
scripts/generate_chains.sh
```

Then, run

```bash
scripts/incremental_answers.sh
```

Then use `notebooks/figures_aime.ipynb` to recreate the plots from the paper.

## Citing
If you find our paper or code useful, consider citing us:
```bibtex
@misc{jurayj2025finalanswertesttimescaling,
      title={Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering}, 
      author={William Jurayj and Jeffrey Cheng and Benjamin Van Durme},
      year={2025},
      eprint={2502.13962},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.13962}, 
}
```
