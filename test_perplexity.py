import os
import torch
import time
import numpy as np
import argparse
import data
from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
from torch.utils.data import DataLoader, DistributedSampler
from model import utils as mutils

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

def simple_score_entropy(graph, score, sigma, x, x0):
    if graph.graph_type=='absorb':
        rel_ind = x == graph.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term
        return entropy
    elif graph.graph_type=='uniform':
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - graph.dim / (esigm1 + graph.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / graph.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / graph.dim
        return pos_term - neg_term
    elif graph.graph_type=='roulette':
        g=1-graph.p_m

        sg = torch.expm1(sigma*g)
        sm = torch.expm1(sigma*graph.p_m)

        r_ba=sg/(sm * torch.exp(sigma*g) * (graph.dim-1))
        r_ca = torch.exp(-sigma*g)*(1+sg/(graph.dim-1))/sm
        r_bc = sg/(torch.exp(sigma*g)+graph.dim-2)#/torch.exp(sigma)

        r_cb = 1/r_bc

        # negative term
        score = torch.scatter(score, -1, (x*0+graph.dim-1)[..., None], torch.zeros_like(score[..., :1]))

        neg_term = score.sum(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1)

        neg_term = torch.where(
            x==(graph.dim-1),
            r_ba * neg_term+torch.gather(score, -1, x0[..., None]).squeeze(-1)*(r_ca-r_ba),
            neg_term
        )
        neg_term = torch.where(
            x==x0,
            r_bc * neg_term,
            neg_term
        )
        neg_term = torch.where(
            torch.logical_and(x != 50257, x != x0),
            neg_term+torch.gather(score, -1, x0[..., None]).squeeze(-1)*(r_cb-1) ,
            neg_term
        )

        #positive term
        sexp = score.exp()
        sexp = torch.scatter(sexp, -1, (x*0+graph.dim-1)[..., None], torch.zeros_like(sexp[..., :1]))
        pos_term = sexp.sum(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) 
        loss = (pos_term - neg_term )
        q = ((x==graph.dim-1)*graph.p_m)   +   ((x!=graph.dim-1)*(1-graph.p_m)/(graph.dim-1))
        return loss*q
    else: 
        print('Error: Graph type not implemented.')

def probs_to_score(graph, score_fn, x, sigma, dsigma):
    sigma = sigma[..., None]
    p_m = graph.p_m
    dim = graph.dim
    graph_type = graph.graph_type        
    output = score_fn(x, sigma)
    f = F.softmax(output, dim=2)

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
    elif graph_type=='uniform':
        mod_sigma = sigma.clone()
        mod_mask = mod_sigma < 0.0015
        mod_sigma[mod_mask] = 0.0015
        sg = torch.expm1(mod_sigma)
        r_bc = sg/(torch.exp(mod_sigma)+dim-1)
        r_cb = 1/r_bc
        score = (1+f*(r_cb[..., None]-1)+torch.gather(f, -1, x[..., None])*(r_bc[..., None]-1)).squeeze()
    elif graph_type=='absorb':
        score = f/(torch.expm1(sigma)[..., None])
    return score

def J0(args, train_iter, device, noise, graph, model):
        sampling_eps=0.0001
        losses = []
        loss_type = graph.loss_type
        for k in range(args.no_datapoints//args.batch_size):
            batch = next(train_iter)['input_ids']
            for i in range(args.batch_size):
                minibatch = batch[i:i+1].repeat(args.batch_size, 1).to(device)
                t = torch.linspace(sampling_eps, (1 - sampling_eps), 1024)
                total_sigma, total_dsigma = noise(t)
                loss_i = []
                for j in range(1024//args.batch_size):
                    sigma, dsigma = total_sigma[j*args.batch_size:(j+1)*args.batch_size].to(device), total_dsigma[j*args.batch_size:(j+1)*args.batch_size].to(device)
                    perturbed_minibatch = graph.sample_transition(minibatch, sigma[:, None])
                    log_score_fn = mutils.get_score_fn(model, train='True', sampling=False)
                    if loss_type=='cedd':
                        score = probs_to_score(graph, log_score_fn, perturbed_minibatch, sigma, dsigma)
                        score.scatter_(-1, perturbed_minibatch[..., None], torch.ones_like(score))
                        log_score = score.log()
                    elif loss_type=='sedd':
                        log_score = log_score_fn(perturbed_minibatch, sigma)
                    loss_ij = graph.score_entropy(log_score, sigma[:, None], perturbed_minibatch, minibatch)
                    loss_ij = (dsigma[:, None] * loss_ij).sum(dim=-1).squeeze().mean()
                    loss_i.append(loss_ij.cpu().numpy())
                losses = losses+loss_i
                np.save(args.model_path+'_eval/'+'lossesJ0_'+'dataset_'+args.dataset+'_length_'+str(args.length)+'.npy', np.array(losses)/args.length)
            print('k: ', k)
            print(np.exp(np.array(losses).mean()/args.length))

def J1(args, train_iter, device, noise, graph, model):
        sampling_eps=0.0001
        losses = []
        loss_type = graph.loss_type
        for i in range(args.no_batches):
            batch = next(train_iter)['input_ids'].to(device)
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            sigma, dsigma = noise(t)
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

            log_score_fn = mutils.get_score_fn(model, train='True', sampling=False)
            if loss_type=='cedd':
                score = probs_to_score(graph, log_score_fn, perturbed_batch, sigma, dsigma)
                score.scatter_(-1, perturbed_batch[..., None], torch.ones_like(score))
                log_score = score.log()
            elif loss_type=='sedd':
                log_score = log_score_fn(perturbed_batch, sigma)       
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss).sum(dim=-1)
            mean_loss = loss.mean()
            losses.append(mean_loss.cpu().numpy())
            if i%100==0:
                print('iter:', i)
                print('mean:', np.exp(np.array(losses).mean()/args.length))
            if i%1000==0:
                np.save(args.model_path+'_eval/'+'lossesJ1_'+'dataset_'+args.dataset+'_length_'+str(args.length)+'.npy', np.array(losses)/args.length)


def J2(args, train_iter, device, noise, graph, model):
        sampling_eps=0.0001
        losses = []
        loss_type = graph.loss_type
        for i in range(args.no_batches):
            
            batch = next(train_iter)['input_ids'].to(device)
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            sigma, dsigma = noise(t)
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

            log_score_fn = mutils.get_score_fn(model, train='True', sampling=False)
            if loss_type=='cedd':
                score = probs_to_score(graph, log_score_fn, perturbed_batch, sigma, dsigma)
                score.scatter_(-1, perturbed_batch[..., None], torch.ones_like(score))
                log_score = score.log()
            elif loss_type=='sedd':
                log_score = log_score_fn(perturbed_batch, sigma)       
            loss = simple_score_entropy(graph, log_score, sigma[:, None], perturbed_batch, batch)

            if graph.graph_type=='uniform':
                loss = (dsigma[:, None] * loss-(1-1/graph.dim)*dsigma[:, None]).sum(dim=-1)
                mean_loss = loss.mean()
                losses.append(mean_loss.cpu().numpy())
                if i%100==0:
                    print('iter:', i)
                    print('mean:', np.exp(np.array(losses).mean()/args.length+np.log(50257)) )
                if i%100==0:
                    np.save(args.model_path+'_eval/'+'lossesJ2_'+'dataset_'+args.dataset+'_length_'+str(args.length)+'.npy', np.array(losses)/args.length+np.log(50257))
            elif graph.graph_type=='roulette':
                loss = (dsigma[:, None] * loss).sum(dim=-1)
                mean_loss = loss.mean()
                losses.append(mean_loss.cpu().numpy())
                if i%100==0:
                    print('iter:', i)
                    print('mean:', np.exp(np.array(losses).mean()/args.length + (1-(1-graph.p_m)/(graph.dim-1))*(sampling_eps-1)/graph.p_m ))
                if i%100==0:
                    np.save(args.model_path+'_eval/'+'lossesJ2_'+'dataset_'+args.dataset+'_length_'+str(args.length)+'.npy', np.array(losses)/args.length+ (1-(1-graph.p_m)/(graph.dim-1))*(sampling_eps-1)/graph.p_m )
            elif graph.graph_type=='absorb':
                loss = (dsigma[:, None] * loss).sum(dim=-1)
                mean_loss = loss.mean()
                losses.append(mean_loss.cpu().numpy())
                if i%100==0:
                    print('iter:', i)
                    print('mean:', np.exp(np.array(losses).mean()/args.length + (sampling_eps-1) ) )
                if i%100==0:
                    np.save(args.model_path+'_eval/'+'lossesJ2_'+'dataset_'+args.dataset+'_length_'+str(args.length)+'.npy', np.array(losses)/args.length+ (sampling_eps-1))


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="exp_local/openwebtext/x", type=str)
    parser.add_argument("--dataset", default="lm1b", type=str)
    parser.add_argument("--length", default=128, type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--perturbed_points_nr", type=int, default=128*(50*1024))#J1 and J2: batchsize times number of batches #J0: total number of scores evaluated at x_\tau#batchsize, times number of batches, times 1024
    parser.add_argument("--cache_dir", type=str, default='data')
    parser.add_argument("--J", type=str, default='2')
    args = parser.parse_args()
    try:
        args.no_batches = args.perturbed_points_nr//args.batch_size#relevant for J1 and J2
        args.no_datapoints = args.perturbed_points_nr//1024#relevant for J0
    except:
        print('args.perturbed_points_nr should be bigger than args.batch_size')
    device = torch.device('cuda')

    create_directory_if_not_exists(args.model_path+'_eval/')
    train_set = data.get_dataset(args.dataset, "train", cache_dir=args.cache_dir, block_size=args.length)

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,   # Enable shuffling of the data
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    ))
    train_iter = iter(train_loader)

    print(f"The size of the dataset: {len(train_set)}")

    with torch.no_grad():
        
        model, graph, noise = load_model(args.model_path, device)
        if args.J=='0':
            J0(args, train_iter, device, noise, graph, model)
        elif args.J=='1':
            J1(args, train_iter, device, noise, graph, model)
        elif args.J=='2':
            J2(args, train_iter, device, noise, graph, model)
if __name__=="__main__":
    main()
