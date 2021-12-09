import random
import torch
import logging
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'refine', 'translate', 'concode', 'clone', 'defect'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--add_lang_ids", action='store_true')
    # plbart unfinished
    parser.add_argument("--model_name", default="roberta",
                        type=str, choices=['roberta', 'codebert', 'graphcodebert', 'bart', 'plbart', 't5', 'codet5'])
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")  # previous one 42
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--cache_path", type=str, default='cache_data')
    parser.add_argument("--res_dir", type=str, default='results',
                        help='directory to save fine-tuning results')
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--model_dir", type=str, default='saved_models',
                        help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard',
                        help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1,
                        help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0,
                        help='index of the gpu to use in a cluster')
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='outputs', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--add_task_prefix", action='store_true',
                        help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true',
                        help="Whether to evaluate bleu on dev set.")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    args = parser.parse_args()
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_count = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_count)
    args.device = device
    args.cpu_count = cpu_count


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_hyperparas(args):
    # the hyperparameters are same to original setting in open-source codes
    if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
        args.lr = 5e-5
        args.adam_epsilon = 1e-8
        args.gradient_accumulation_steps = 1
        args.weight_decay = 0.0
        args.batch_size = 48
        args.beam_size = 10
        args.max_source_length = 256
        args.max_target_length = 128
        args.warmup_steps = 0

        if args.task == 'summarize':
            if args.sub_task == 'ruby':
                args.eval_steps = 400
                args.train_steps = 20000
            elif args.sub_task == 'javascript':
                args.eval_steps = 600
                args.train_steps = 30000
            else:
                args.eval_steps = 1000
                args.train_steps = 50000
