# Training

To train: python train.py

# Evaluation of the canonical reverse flow

1) To evaluate the generative perplexity: python test_genperplexity.py --mode gen
then after it finishes running, run: test_genperplexity.py --mode gpt

2) To evaluate the perplexity bound: python test_perplexity_flow.py --dataset DATASET
 
# Evaluation of the corrector reverse flow

1) To evaluate the generative perplexity: python test_genperplexity_corrector.py --mode gen
then after it finishes running, run: test_genperplexity_corrector.py --mode gpt

2) To evaluate the perplexity bound: python test_perplexity_flow_corrector.py --dataset DATASET
