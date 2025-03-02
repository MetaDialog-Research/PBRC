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

def J1(args, train_iter, device, noise, graph, model):
        sampling_eps=0.0001
        losses = []
        loss_type = graph.loss_type
        for i in range(args.no_batches):
            #print(i)
            
            databatch = next(train_iter)['input_ids'].to(device)
            B, L = databatch.shape
            sourcebatch = graph.sample_limit(databatch.shape).to(databatch)
            
            t = (1 - 2*sampling_eps) * torch.rand(databatch.shape[0], device=databatch.device)+sampling_eps
            kt=t
            kt_d =t*0+1
            #kt=t**2
            #kt_d =2*t
            perturbed_batch = graph.sample_transition(sourcebatch, databatch, t[:, None])

            mask= perturbed_batch!=databatch

            log_score_fn = mutils.get_score_fn(model, train='True', sampling=False)
            log_score = log_score_fn(perturbed_batch, t)

            probs = torch.softmax(log_score, dim=-1)

            loss1 = mask * (torch.log( torch.gather(probs, dim=-1, index=databatch.unsqueeze(-1) ).squeeze()+0.0000000001)     +      1)
            loss1 = -kt_d[:, None]*loss1/(1-kt[:, None])

            loss2 = 1 - torch.gather(probs, dim=-1, index=perturbed_batch.unsqueeze(-1)).squeeze()
            loss2 = kt_d[:, None]*loss2/(1-kt[:, None])

            loss = loss1 + loss2

            loss =  loss.sum(-1).mean()
            
            losses.append(loss.cpu().numpy())
            
            if torch.isinf(torch.tensor(np.exp(np.array(losses).mean()/args.length))).item():
                break
            if i%100==0:
                print('iter:', i)
                print('mean:', np.exp(np.array(losses).mean()/args.length))
            if i%1000==0:
                np.save(args.model_path+'_eval/'+'lossesJ1_'+'dataset_'+args.dataset+'_length_'+str(args.length)+'.npy', np.array(losses)/args.length)



def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="exp_local/openwebtext/x", type=str)
    parser.add_argument("--dataset", default="ptb", type=str)
    parser.add_argument("--length", default=128, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--perturbed_points_nr", type=int, default=128*(50*1024))#J1 and J2: batchsize times number of batches #J0: total number of scores evaluated at x_\tau#batchsize, times number of batches, times 1024
    parser.add_argument("--cache_dir", type=str, default='data')
    parser.add_argument("--J", type=str, default='1')
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
        J1(args, train_iter, device, noise, graph, model)
if __name__=="__main__":
    main()
