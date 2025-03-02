import abc
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import matplotlib.pyplot as plt
import os
import numpy as np
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise, visualize):
        super().__init__()
        self.graph = graph
        self.noise = noise
        self.visualize = visualize
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.i = 0

    def probs_to_score(self, score_fn, x, sigma, dsigma):
        p_m = self.graph.p_m
        dim = self.graph.dim
        graph_type = self.graph.graph_type        
        score = score_fn(x, sigma).log()
        score = F.softmax(score, dim=2)

        if graph_type=='roulette':
            g=1-p_m
            sg = torch.expm1(sigma*g)
            sm = torch.expm1(sigma*p_m)
            r_ba=sg/(sm * torch.exp(sigma*g) * (dim-1))
            r_ca = torch.exp(-sigma*g)*(1+sg/(dim-1))/sm

            mod_sigma = sigma.clone()
            mod_mask = mod_sigma < 0.5
            mod_sigma[mod_mask] = (mod_sigma[mod_mask] * 1.1 + 1.1).log()
            sg = torch.expm1(mod_sigma*g)
            sm = torch.expm1(mod_sigma*p_m)
            r_bc = sg/(torch.exp(mod_sigma*g)+dim-2)
            r_cb = 1/r_bc

            score = torch.where(x.unsqueeze(-1)==(dim-1),
            (r_ba[..., None]+score*(r_ca[..., None]-r_ba[..., None])).squeeze(),
            (1+score*(r_cb[..., None]-1)+torch.gather(score, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
            )
        elif graph_type=='uniform':
            mod_sigma = sigma.clone()
            mod_mask = mod_sigma < 0.003
            mod_sigma[mod_mask] = 0.003
            sg = torch.expm1(mod_sigma)
            r_bc = sg/(torch.exp(mod_sigma)+dim-1)
            r_cb = 1/r_bc
            score = (1+score*(r_cb[..., None]-1)+torch.gather(score, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
        elif graph_type=='absorb':
            score = score/(torch.expm1(sigma)[..., None])
        return score

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        loss_type = self.graph.loss_type
        sigma, dsigma = self.noise(t)

        if loss_type=='cedd' or loss_type=='re_sedd':
            score = self.probs_to_score(score_fn, x, sigma, dsigma)
        elif loss_type=='sedd':
            score = score_fn(x, sigma)
            

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma
        loss_type = self.graph.loss_type

        if loss_type=='cedd' or loss_type=='re_sedd':
            score = self.probs_to_score(score_fn, x, curr_sigma, dsigma)
            score.scatter_(-1, x[..., None], torch.ones_like(score))
        elif loss_type=='sedd':
            score = score_fn(x, curr_sigma)
        

        stag_score = self.graph.staggered_score(score, dsigma)
        
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def probs_to_score(self, score_fn, x, sigma, dsigma):
        p_m = self.graph.p_m
        dim = self.graph.dim
        graph_type = self.graph.graph_type        
        score = score_fn(x, sigma).log()
        score = F.softmax(score, dim=2)

        if graph_type=='roulette':
            g=1-p_m
            sg = torch.expm1(sigma*g)
            sm = torch.expm1(sigma*p_m)
            r_ba=sg/(sm * torch.exp(sigma*g) * (dim-1))
            r_ca = torch.exp(-sigma*g)*(1+sg/(dim-1))/sm

            mod_sigma = sigma.clone()
            mod_mask = mod_sigma < 0.5
            mod_sigma[mod_mask] = (mod_sigma[mod_mask] * 1.1 + 1.1).log()
            sg = torch.expm1(mod_sigma*g)
            sm = torch.expm1(mod_sigma*p_m)
            r_bc = sg/(torch.exp(mod_sigma*g)+dim-2)
            r_cb = 1/r_bc

            score = torch.where(x.unsqueeze(-1)==(dim-1),
            (r_ba[..., None]+score*(r_ca[..., None]-r_ba[..., None])).squeeze(),
            (1+score*(r_cb[..., None]-1)+torch.gather(score, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
            )
        elif graph_type=='uniform':
            mod_sigma = sigma.clone()
            mod_mask = mod_sigma < 0.003
            mod_sigma[mod_mask] = 0.003
            sg = torch.expm1(mod_sigma)
            r_bc = sg/(torch.exp(mod_sigma)+dim-1)
            r_cb = 1/r_bc
            score = (1+score*(r_cb[..., None]-1)+torch.gather(score, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
        elif graph_type=='absorb':
            score = score/(torch.expm1(sigma)[..., None])
        return score
    
    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]
        sigma, dsigma = self.noise(t)
        loss_type = self.graph.loss_type

        if loss_type=='cedd' or loss_type=='re_sedd':
            score = self.probs_to_score(score_fn, x, sigma, dsigma)
            score.scatter_(-1, x[..., None], torch.ones_like(score))
        elif loss_type=='sedd':
            score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    
def sample_sequence_batch(model, tokenizer, length, device, batch_size=1):
    """
    Generates a batch of sequences of length `L` using your custom GPT-2 model unconditionally.
    
    Args:
    - model: Your custom GPT-2 model.
    - tokenizer: Your custom tokenizer.
    - length: The length of the sequences to generate.
    - batch_size: The number of sequences to generate at once.
    - device: The device to run the model on ('cpu' or 'cuda').

    Returns:
    - generated_texts: A list of generated sequences as strings.
    """
    
    model.eval()
    
    # Initialize the input with the start token for each sequence in the batch
    input_ids = [[50257]] * batch_size  # Start with the GPT-2 start token for each sequence
    
    # Convert input to tensor and move to the correct device (CPU or GPU)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).to(device)
    
    # Set model to evaluation mode (no gradients, dropout disabled)
    generated = input_tensor
    with torch.no_grad():
        for _ in range(length):
            # Forward pass through the model to get logits (next token probabilities)
            outputs = model(generated)
            logits = outputs[:, -1, :]  # Take the logits of the last token for each sequence
            
            # Sample from the logits (probability distribution) to pick the next token
            probs = torch.softmax(logits, dim=-1)
            print(probs.squeeze().sort()[0])
            print((probs.squeeze()**2).sum())
            break
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the predicted token to each sequence in the batch
            generated = torch.cat((generated, next_token), dim=1)
    

    return generated[:, 1:]


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, visualize='False', denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise, visualize)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        x = sample_sequence_batch(model, tokenizer, batch_dims[1], device, batch_dims[0])

               
        return x
    
    return pc_sampler

