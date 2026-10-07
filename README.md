# Efficient Perplexity Bound and Ratio Matching in Discrete Diffusion Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official PyTorch implementation of the paper [Efficient Perplexity Bound and Ratio Matching in Discrete Diffusion Language Models](https://openreview.net/forum?id=Mri9WIfxSm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)), ICLR 2025, by [Etrit Haxholli](https://github.com/ehaxholli), [Yeti Z. Gurbuz](https://github.com/yetigurbuz), [Oğul Can](https://github.com/canogulcan) and [Eli Waxman](https://github.com/EliBraginskiy).

[MetaDialog](https://www.metadialog.com/), [MetaDialog Research](https://github.com/MetaDialog-Research) 


# Training and options:

To train: python train.py graph.loss_type=cedd graph.type=roulette graph.p_m=0.95 model=small model.scale=True


a) graph types: absorb, uniform, roulette

b) losses: cedd (this includes cedd*, see e) below); sedd

c) in case of graph.type==roulette then chose probability of masking 'p_m' (from 0 to 1).

d) In case that graph.loss_type==sedd, choose whether you want to scale the output of the neural network

e) One can choose whether to use CEDD or CEDD* by modifying line 88 in losses.py

# Sampling and options:

To sample: python run_sample.py --model_path MODEL_PATH --steps STEPS

# Evaluation:

To evaluate the model in terms of generative perplexity: test_genperplexity.py --model_path MODEL_PATH --steps STEPS --length LENGTH --mode MODE

To evaluate the model using the perplexity bound: test_perplexity.py --model_path MODEL_PATH --J J_BOUND_TYPE


**Mode can be:**

i) 'gen', which generates and saves samples.

ii) 'gpt', which uses gpt2-large to evaluate the generated samples in i).

iii) 'llama', which uses Llama-3.1-8B to evaluate the generated samples in i).

**J can be:**

i) '0', which will evaluate the bound on each point and then average

ii) '1', will use J1 in the paper

iii) '2' will use our J2 from the paper.

# Other Experiments:

'DFM' contains code related to Discrete Flow Matching

'spell_check' contains code related to spelling correction

'GPT2' contains a GPT2-like model implementation

# Acknowledgements

This repository is built upon the foundation provided by https://github.com/louaaron/Score-Entropy-Discrete-Diffusion.

