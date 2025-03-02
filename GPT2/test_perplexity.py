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


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="exp_local/openwebtext/x", type=str)
    parser.add_argument("--dataset", default="lambada", type=str)
    parser.add_argument("--length", default=1024, type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_batches", type=int, default=200)
    parser.add_argument("--cache_dir", type=str, default='data')
    parser.add_argument("--J", type=str, default='1')
    parser.add_argument("--exponentiate", type=bool, default=True)

    args = parser.parse_args()

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
        eval_model, graph, noise = load_model(args.model_path, device)
        total_perplexity = []
        for i in range(args.no_batches):
            print(i)
            s = next(train_iter)['input_ids'].to(device)
            #print('s shape:', s.shape)
            spec_tok = 50257*torch.ones(s.shape[0])
            #print('spec_tok shape:', spec_tok.shape)
            input_batch = torch.cat( (spec_tok.reshape(-1,1).to(s), s[:,:-1]), 1)
            #print('input_batch shape:', input_batch.shape)
            
            #print(i)
            logits = eval_model(input_batch)
            #print('logits shape:', logits.shape)
            perplexity = F.cross_entropy(logits.reshape(-1, logits.size(-1)), s.reshape(-1), reduction='none')
            #print('perplexity shape:', perplexity.shape)
            perplexity = perplexity.reshape(-1, args.length).mean(-1).exp().to(torch.float64).cpu().numpy().tolist()
            total_perplexity += perplexity    
            print(np.exp(np.log(np.array(total_perplexity)).mean()))
            np.save(args.model_path+'_eval/'+'Perplexity_'+'dataset_'+args.dataset+'totalpoints_'+str(args.no_batches*args.batch_size)+'_length_'+str(args.length)+'.npy', np.array(total_perplexity))
            if i>(len(train_set)//args.batch_size):
                break

if __name__=="__main__":
    main()
