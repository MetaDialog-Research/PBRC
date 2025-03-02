import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils

def probs_to_score(graph, score_fn, x, sigma, dsigma):
    sigma = sigma[..., None]
    p_m = graph.p_m
    dim = graph.dim
    graph_type = graph.graph_type        
    score = score_fn(x, sigma)
    f = F.softmax(score, dim=2)

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
        (r_ba[..., None]+f*(r_ca[..., None]-r_ba[..., None])).squeeze(),
        (1+f*(r_cb[..., None]-1)+torch.gather(f, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
        )
        score[score==0] = 0.000000000000001#for stability
    elif graph_type=='uniform':
        mod_sigma = sigma.clone()
        mod_mask = mod_sigma < 0.003
        mod_sigma[mod_mask] = 0.003
        sg = torch.expm1(mod_sigma)
        r_bc = sg/(torch.exp(mod_sigma)+dim-1)
        r_cb = 1/r_bc
        score = (1+f*(r_cb[..., None]-1)+torch.gather(f, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
    elif graph_type=='absorb':
        score = score/(torch.expm1(sigma)[..., None])
    return score

def get_loss_fn(noise, graph, train, config, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        

        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])


        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)

        if config.graph.loss_type == 'cedd':

            log_score = log_score.view(-1, log_score.shape[-1]) 
            batch = batch.view(-1)  
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            if config.graph.type == 'uniform' or config.graph.type == 'roulette' :
                loss = criterion(log_score, batch)
                loss = loss.view(perturbed_batch.shape[0], perturbed_batch.shape[1])
            elif config.graph.type == 'absorb': # since unmasked tokens cannot change in the reverse process, we do not calculate the loss corresponding to these positions.
                dim = config.tokens+1
                mask = perturbed_batch==(dim-1)
                mask = (mask*1)
                loss = criterion(log_score, batch)
                loss = loss.view(perturbed_batch.shape[0], perturbed_batch.shape[1])*mask  
            else:
                print('graph type provided in config has not been implemented')
            loss = (loss*(np.e+0.3/t[:, None]).log()).sum(dim=-1)#(loss).sum(dim=-1)#(loss*(np.e+0.3/t[:, None]).log()).sum(dim=-1)#
            return loss            
        elif config.graph.loss_type == 're_sedd':
            score = probs_to_score(graph, log_score_fn, perturbed_batch, sigma, dsigma)
            score.scatter_(-1, perturbed_batch[..., None], torch.ones_like(score))
            loss = graph.re_score_entropy(score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss).sum(dim=-1)
            return loss

        elif config.graph.loss_type == 'sedd':
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss).sum(dim=-1)
            return loss
        
        else: 
            print('loss type provided in config has not been implemented')

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, config):
    loss_fn = get_loss_fn(noise, graph, train, config)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn
