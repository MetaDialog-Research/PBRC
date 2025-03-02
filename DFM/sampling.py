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

        logits = score_fn(x, t).log()
        probs = torch.softmax(logits, 2)

        rev_rate =  (probs-F.one_hot(x, num_classes=probs.shape[-1]).to(probs))* (step_size *    1/(1-t[..., None]) )  
        #rev_rate =  (probs-F.one_hot(x, num_classes=probs.shape[-1]).to(probs))* (2*step_size *    (t[..., None]/(1-t[..., None]**2) )       )
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x
    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise
    
    def update_fn(self, score_fn, x, t):
        logits = score_fn(x, t).log()
        probs = torch.softmax(logits, 2)

        probs = F.one_hot(x, num_classes=probs.shape[-1]).to(probs) + 2*(probs-F.one_hot(x, num_classes=probs.shape[-1]).to(probs))/(1+t[..., None])        
        #probs = F.one_hot(x, num_classes=probs.shape[-1]).to(probs) + 2*t[..., None]*(probs-F.one_hot(x, num_classes=probs.shape[-1]).to(probs))/(1+t[..., None])        

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
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, visualize='False', denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise, visualize)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(batch_dims).to(device)
        timesteps = torch.linspace(eps, 1-eps, steps + 1, device=device)
        dt = 1 / steps

        for i in range(steps):
            predictor.i+=1
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)


        # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # text_samples = tokenizer.batch_decode(x)
        # print(text_samples)
        # print('-'*100)


        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
        predictor.i=0
               
        return x
    
    return pc_sampler

