import torch
import argparse
import os
from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F
import sampling
import numpy as np
import time
import gc

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="exp_local/openwebtext/x", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--no_batches", type=int, default=480)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--mode", type=str, default='gen')
    args = parser.parse_args()

    device = torch.device('cuda')

    # model_id = "meta-llama/Llama-3.1-8B"

    # tokenizer_llama = AutoTokenizer.from_pretrained(model_id)
    # tokenizer_llama.pad_token = tokenizer_llama.eos_token

    tokenizer_gpt = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token




    with torch.no_grad():
        create_directory_if_not_exists(args.model_path+'_eval/')
        
        if args.mode=='gen':
            model, graph, noise = load_model(args.model_path, device)
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (args.batch_size, args.length), 'analytic', args.steps, device=device
            )


            all_samples = []
            for i in range(args.no_batches):
                print(i)
                s = sampling_fn(model)
                s = s.detach().cpu()
                all_samples.append(s.to(device))
                torch.save(torch.stack(all_samples).reshape(-1, args.length), args.model_path+'_eval/all_samples.pth')





        elif args.mode=='gpt':
            all_samples = torch.load(args.model_path+'_eval/all_samples.pth')
            all_samples = torch.chunk(all_samples, all_samples.shape[0]//8)
            total_perplexity = []
            eval_gpt = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
            i=0
            for s in all_samples:
                #GPT
                i+=1
                print(i)
                _, logits = eval_gpt(s, labels=s)[:2]
                perplexity = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), s[..., 1:].reshape(-1), reduction='none')
                perplexity = perplexity.reshape(-1, args.length-1).mean(-1).exp().to(torch.float64).cpu().numpy().tolist()
                total_perplexity += perplexity 
                print('GPT2 perplexity:', np.array(total_perplexity).mean())
                np.save(args.model_path+'_eval/'+'GenPerplexity_'+'totalpoints_'+str(args.no_batches*args.batch_size)+'_length_'+str(args.length)+'.npy', np.array(total_perplexity))


        elif args.mode=='llama':
            all_samples = torch.load(args.model_path+'_eval/all_samples.pth')
            all_samples = torch.chunk(all_samples, all_samples.shape[0]//8)
            total_llama_perplexity = []
            eval_llama = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16).to(device).eval()
            with torch.inference_mode():
                i=0
                for s in all_samples:
                    #LLAMA
                    i+=1
                    print(i)
                    text_samples = tokenizer_gpt.batch_decode(s)
                    #print('s:', s)
                    #print(text_samples)
                    #print('GPT2 perplexity:', np.array(total_perplexity).mean())
                    encoded_inputs = tokenizer_llama(
                        text_samples,
                        return_tensors="pt",  # PyTorch tensor output
                        padding=True,          # Pads all sequences to the same length
                        #truncation=True        # Truncates sequences to the model's max length if needed
                    )
                    # Extract input_ids and attention_mask
                    input_ids = encoded_inputs['input_ids'].to(device)
                    attention_mask = encoded_inputs['attention_mask'].to(device)
                    #print('input_ids.shape:', input_ids.shape)
                    #print('attention_mask.shape:', attention_mask.shape)
                    #print(attention_mask)

                    # Shift input_ids by one to create labels and set padding tokens to -100
                    labels = input_ids.clone()
                    labels[labels == tokenizer_llama.pad_token_id] = 50000  # Ignore padding in the loss calculation
                    #print('labels.shape:', labels.shape)
                    labels = labels.to(device)

                    # Get model outputs
                    outputs = eval_llama(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    
                    llama_logits = outputs.logits
                    #print('llama_logits.shape:', llama_logits.shape)
                    #print('logits:', torch.softmax(logits,-1).sum(-1))

                    # Calculate the loss for each token in the batch without reduction
                    llama_perplexity = F.cross_entropy(llama_logits[:, :-1].reshape(-1, llama_logits.size(-1)), labels[..., 1:].reshape(-1), reduction='none')
                    #print('llama_perplexity.shape:', llama_perplexity.shape)
                    llama_perplexity = llama_perplexity.reshape(-1, input_ids.shape[1]-1)
                    #print('llama_perplexity.shape:', llama_perplexity.shape)
                    llama_perplexity = llama_perplexity*attention_mask[..., 1:]
                    llama_perplexity = llama_perplexity.sum(-1)/attention_mask[..., 1:].sum(-1)
                    
                    llama_perplexity=llama_perplexity.exp().to(torch.float64).cpu().numpy().tolist()

                    total_llama_perplexity += llama_perplexity 
                    np.save(args.model_path+'_eval/'+'LlamaGenPerplexity_'+'totalpoints_'+str(args.no_batches*args.batch_size)+'_length_'+str(args.length)+'.npy', np.array(total_llama_perplexity))
                    del llama_logits, outputs, input_ids, attention_mask, labels
                    torch.cuda.empty_cache()
                    print('Llama perplexity:', np.array(total_llama_perplexity).mean())


if __name__=="__main__":
    main()