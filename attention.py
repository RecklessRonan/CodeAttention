from utils import (get_filenames, get_elapse_time,
                   load_and_cache_gen_data, get_ast_nx, format_attention, num_layers,
                   index_to_code_token, format_special_chars)
from models import bulid_or_load_gen_model
from configs import add_args, set_dist, set_seed, set_hyperparas


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
from tree_sitter import Language, Parser
import networkx as nx
import numpy as np
import sys
sys.setrecursionlimit(1500)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_attention_and_subtoken(args, data, examples, model, tokenizer):
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.attention_batch_size,
                            num_workers=4, pin_memory=True)
    model.eval()
    attention_list = []
    subtokens_list = []
    logger.info("Obtain subtokens and their attention")
    for batch in tqdm(dataloader, total=len(dataloader), desc="Computing attention"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        for source_id in source_ids:
            subtokens = tokenizer.convert_ids_to_tokens(source_id)
            subtokens_list.append(subtokens)

        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
                _, _, _, attention = model(source_ids=source_ids, source_mask=source_mask,
                                           target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                attention = outputs.encoder_attentions
        includer_layers = list(range(num_layers(attention)))
        attention = format_attention(attention, layers=includer_layers[-1])
        attention = attention.detach().cpu().numpy()
        attention_list.append(attention)
    attention_list = np.concatenate(attention_list, axis=0)
    print('attention_list shape: ', attention_list.shape)
    print('subtokens_list length: ', len(subtokens_list))
    print('subtokens 0: ')
    print(subtokens_list[0])
    return attention_list, subtokens_list


def number_subtoken(subtokens_list, tokens_list, tokenizer):
    """[summary]

    Args:
        subtokens_list ([type]): [description]
        tokens_list ([type]): [description]
    """

    assert len(subtokens_list) == len(tokens_list)
    subtoken_numbers_list = []
    for i in range(len(subtokens_list)):
        subtokens = subtokens_list[i]
        token_numbers = list(tokens_list[i].keys())
        tokens = list(tokens_list[i].values())
        assert len(token_numbers) == len(tokens)

        subtoken_numbers = []
        subtokens = format_special_chars(subtokens)
        for j in range(len(subtokens)):
            pos = 0

            if subtokens[j] in tokenizer.additional_special_tokens:
                # the special tokens of tokenizer is not involved in AST tree, we use -1 to tag it
                subtoken_numbers.append(-1)
            else:
                if subtokens[j] in tokens[pos]:
                    subtoken_numbers.append(pos)
                else:
                    pos += 1
        subtoken_numbers_list.append(subtoken_numbers)

    print('subtoken_numbers_list length: ', len(subtoken_numbers_list))
    print('subtoken_numbers_list 0: ')
    print(subtoken_numbers_list[0])
    return subtoken_numbers_list


def get_ast_and_token(examples, parser, lang):
    ast_list = []
    tokens_list = []

    logger.info("Parse AST trees and obtain leaf tokens")

    i = 0
    for example in tqdm(examples):
        ast_example = get_ast_nx(example, parser, lang)
        G = ast_example.ast
        ast_list.append(G)
        T = nx.dfs_tree(G, 0)
        leaves = [x for x in T.nodes() if T.out_degree(x) ==
                  0 and T.in_degree(x) == 1]
        tokens_dict = {}
        for leaf in leaves:
            feature = G.nodes[leaf]['features']
            if feature.type != 'comment':
                start = feature.start_point
                end = feature.end_point
                token = index_to_code_token([start, end], ast_example.source)
                if i == 0:
                    print('leaf: ', leaf, 'start: ', start,
                          ', end: ', end, ', token: ', token)
                tokens_dict[leaf] = token

        if i == 0:
            print(T.nodes)
            print(T.edges)
            print('raw_code: ', ast_example.source)
            print('leaves: ', leaves)
            i += 1

        tokens_list.append(tokens_dict)
    print('ast list length', len(ast_list))
    print('tokens list length', len(tokens_list))
    print('tokens 0: ')
    print(tokens_list[0])
    return ast_list, tokens_list


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    set_dist(args)
    set_seed(args)
    set_hyperparas(args)

    logger.info(args)

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
        args, args.train_filename, pool, tokenizer, 'attention', is_sample=True, is_attention=True)

    Language.build_library(
        'build/my-language.so',
        [
            '/data/code/tree-sitter/tree-sitter-ruby',
            '/data/code/tree-sitter/tree-sitter-javascript',
            '/data/code/tree-sitter/tree-sitter-go',
            '/data/code/tree-sitter/tree-sitter-python',
            '/data/code/tree-sitter/tree-sitter-java',
            # '/data/code/tree-sitter/tree-sitter-php',
        ]
    )
    language = Language('build/my-language.so', args.sub_task)
    parser = Parser()
    parser.set_language(language)

    ast_list, tokens_list = get_ast_and_token(examples, parser, args.sub_task)
    attention_list, subtokens_list = get_attention_and_subtoken(
        args, data, examples, model, tokenizer)
    subtoken_numbers_list = number_subtoken(
        subtokens_list, tokens_list, tokenizer)


if __name__ == "__main__":
    main()
