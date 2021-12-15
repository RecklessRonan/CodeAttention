import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
import json
import random
import argparse
import multiprocessing
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import os
import logging
from bertviz.util import format_attention, num_layers

from configs import add_args, set_dist, set_seed, set_hyperparas
from models import bulid_or_load_gen_model
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_attention(args, data, examples, model, tokenizer):
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.attention_batch_size,
                            num_workers=4, pin_memory=True)
    model.eval()
    attention_list = []
    for batch in tqdm(dataloader, total=len(dataloader), desc="Computing attention"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
                _, _, _, attention = model(source_ids=source_ids, source_mask=source_mask,
                                           target_ids=target_ids, target_mask=target_mask)

                includer_layers = list(range(num_layers(attention)))
                attention = format_attention(attention, includer_layers)
                print(attention.shape)
                # attention_list.append(attention.detach())
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
    return attention_list


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    set_dist(args)
    set_seed(args)
    set_hyperparas(args)

    if args.task in ['summarize', 'translate']:
        config, model, tokenizer = bulid_or_load_gen_model(args)

    model_dict = os.path.join(
        args.output_dir, 'checkpoint-best-ppl/pytorch_model.bin')
    model.load_state_dict(torch.load(model_dict))

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_count)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    examples, data = load_and_cache_gen_data(
        args, args.train_filename, pool, tokenizer, 'attention', is_sample=True)
    attention_list = get_attention(
        args, data, examples, model, tokenizer)


if __name__ == "__main__":
    main()
