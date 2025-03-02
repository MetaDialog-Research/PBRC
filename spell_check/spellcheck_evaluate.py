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
from torch.utils.data import DataLoader, Dataset


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def mistake_counter(args, train_iter, device, noise, graph, model):
        losses = []
        losses2 = []
        loss_type = graph.loss_type
        for i in range(args.no_batches):
            batch = next(train_iter).to(device)
            print(batch)
            
            t = args.mistake_percentage * torch.ones(batch.shape[0], device=batch.device)
            sigma, dsigma = noise(t)
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])
            mask = (batch!=perturbed_batch)
            
            nr_mistakes = (mask*1).sum().item()
            if nr_mistakes>0:
                print('----------------------')
                print('orig_mist:', nr_mistakes)
                if loss_type=='cedd':
                    print('!'*100)
                    log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
                    log_score = log_score_fn(perturbed_batch, sigma)       
                    preds = log_score.argmax(-1)
                elif loss_type=='sedd':
                    print('@'*100)
                    log_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
                    score = log_score_fn(perturbed_batch, sigma)
                    stag_score = graph.staggered_score(score, sigma[0], sigma[0], perturbed_batch)
                    probs = stag_score * graph.transp_transition(perturbed_batch, sigma[0])
                    preds = probs.argmax(-1)
                

                mask2 = (batch!=preds)
                nr_endmistakes = (mask2*1).sum().item()
                print('nr_endmistakes:', nr_endmistakes)


                preds[~mask] = batch[~mask]
                mask3 = (batch!=preds)
                nr_lingeringmistakes = (mask3*1).sum().item()
                print('nr_lingeringmistakes:', nr_lingeringmistakes)

                losses.append(nr_endmistakes/nr_mistakes)
                losses2.append(nr_lingeringmistakes/nr_mistakes)
                

                print('Average mistakes at the end as percentage:', np.array(losses).mean())
                print('Average lingering mistakes as percentage:', np.array(losses2).mean())


class CharacterLevelDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize text at character level
        tokens = [tokenizer(c, add_special_tokens=False)['input_ids'] for c in text]
        self.tokenized_text = [t[0] for t in tokens if len(t) > 0]  # Flatten and remove empty tokens
        
        # Form subsequences of length `block_size`
        self.subsequences = [
            self.tokenized_text[i:i + block_size]
            for i in range(0, len(self.tokenized_text), block_size)
            if len(self.tokenized_text[i:i + block_size]) == block_size
        ]

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        return torch.tensor(self.subsequences[idx], dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="exp_local/openwebtext/x", type=str)
    parser.add_argument("--length", default=128, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mistake_percentage", type=float, default=0.05)
    parser.add_argument("--no_batches", type=int, default=10000)
    parser.add_argument("--file_path", type=str, default="text_files/pap")
    args = parser.parse_args()
    device = torch.device('cuda')

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Open the file and read its contents
    with open(args.file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Prepare dataset and dataloader
    dataset = CharacterLevelDataset(text, tokenizer, args.length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    train_iter = iter(dataloader)

    with torch.no_grad():
        
        model, graph, noise = load_model(args.model_path, device)
        mistake_counter(args, train_iter, device, noise, graph, model)

    

if __name__=="__main__":
    main()
