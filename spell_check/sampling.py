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
            mod_mask = mod_sigma < 0.0015
            mod_sigma[mod_mask] = 0.0015
            sg = torch.expm1(mod_sigma)
            r_bc = sg/(torch.exp(mod_sigma)+dim-1)
            r_cb = 1/r_bc
            score = (1+score*(r_cb[..., None]-1)+torch.gather(score, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
        elif graph_type=='absorb':
            pass#score = score/(torch.expm1(sigma)[..., None])#for efficiency we move the implementation of division in graph_lib.py for the analytic sampler.
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
            if self.graph.graph_type=='absorb': 
                score = score/(torch.expm1(sigma)[..., None])
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
            if self.graph.graph_type!='absorb':  
                score.scatter_(-1, x[..., None], torch.ones_like(score))
        elif loss_type=='sedd':
            score = score_fn(x, curr_sigma)
        

        stag_score = self.graph.staggered_score(score, dsigma, curr_sigma, x)
        
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return probs.argmax(-1)#sample_categorical(probs)

    
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
            mod_mask = mod_sigma < 0.0015
            mod_sigma[mod_mask] = 0.0015
            sg = torch.expm1(mod_sigma)
            r_bc = sg/(torch.exp(mod_sigma)+dim-1)
            r_cb = 1/r_bc
            score = (1+score*(r_cb[..., None]-1)+torch.gather(score, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
        elif graph_type=='absorb':
            pass#score = score/(torch.expm1(sigma)[..., None])
        return score
    
    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]
        sigma, dsigma = self.noise(t)
        loss_type = self.graph.loss_type

        if loss_type=='cedd' or loss_type=='re_sedd':
            score = self.probs_to_score(score_fn, x, sigma, dsigma)
            if self.graph.graph_type!='absorb':  
                score.scatter_(-1, x[..., None], torch.ones_like(score))
        elif loss_type=='sedd':
            score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma, sigma, x)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return probs.argmax(-1)#sample_categorical(probs)
                       

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
    def pc_sampler(model, perturbed_batch):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = perturbed_batch.to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            predictor.i+=1
            y=x.clone()
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            if visualize:
                mask = (x[0]==y[0]).cpu()
                opposmask = (x[0]!=y[0]).cpu()
                os.makedirs('visual', exist_ok=True)
                plt.scatter(np.arange(x.shape[-1])[mask],x[0][mask].cpu().numpy())
                plt.scatter(np.arange(x.shape[-1])[opposmask],x[0][opposmask].cpu().numpy())
                plt.savefig('visual/'+str(predictor.i)+'.png')
                plt.clf()
            
        y=x.clone()
        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
        if predictor.visualize:
            mask = (x[0]==y[0]).cpu()
            opposmask = (x[0]!=y[0]).cpu()
            os.makedirs('visual', exist_ok=True)
            plt.scatter(np.arange(x.shape[-1])[mask],x[0][mask].cpu().numpy())
            plt.scatter(np.arange(x.shape[-1])[opposmask],x[0][opposmask].cpu().numpy())
            plt.savefig('visual/129.png')
            plt.clf()
        predictor.i=0
        # x_print=x.clone()
        # eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
        # _, logits = eval_model(x_print.squeeze(), labels=x_print.squeeze())[:2]
        # perplexity = F.cross_entropy(logits.squeeze()[:-1], x_print.squeeze()[1:]).exp()
        # print('perplexity:', perplexity)
        # total_perplexity /= batches           
        # with open('perplexities0.001p.txt', 'a') as file:
        #     file.write(f"{perplexity},\n")
               
        return x
    
    return pc_sampler

