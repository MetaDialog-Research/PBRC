# Training

To train: python spellcheck_train.py graph.loss_type=cedd graph.type=roulette graph.p_m=0.95 model=small model.scale=True

# Options:

a) graph types: absorb, uniform, roulette

b) losses: cedd (this includes cedd*, see e) below); sedd

c) In case that graph.loss_type==sedd, choose whether you want to scale the output of the neural network

d) One can choose whether to use CEDD or CEDD* by modifying line 88 in losses.py

# Evaluation

To evaluate correction with a single step from time 0.05: python spellcheck_evaluate.py --model_path MODEL_PATH --steps STEPS

To evaluate correction with multiple steps from time 1: python spellcheck_evaluate_ms.py --model_path MODEL_PATH --steps STEPS
