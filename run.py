import os
import sys
import json
import numpy as np
import random
import nltk
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
np.set_printoptions(threshold=np.inf)

import torch
from ChickenRabbit import ChickenRabbitDataset, eval_split as eval_split_CR
from GCD import GCDDataset, eval_split as eval_split_GCD
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.model_multiplier import GPT
from mingpt.trainer_multiplier import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from itertools import permutations

# -----------------------------------------------------------------------------

def get_config(type = "CR"):
    C = CN()

    # system
    C.system = CN()
    # TODO: random seed for model can be set here
    C.system.init_seed = 0 # will change the weight initialization
    C.system.work_dir = './test'

    # data
    if type == "CR":
        C.data = ChickenRabbitDataset.get_default_config()
    else:
        C.data = GCDDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    
    # trainer
    C.trainer = Trainer.get_default_config()
    if type == "CR":
        C.trainer.task = "ChickenRabbit"
    else: 
        C.trainer.task = "GCD"
    return C

def batch_end_callback(trainer, model, train_dataset, test_dataset, type="CR"):
    if type == "CR":
        eval_split = eval_split_CR
    else:
        eval_split = eval_split_GCD

    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    if trainer.iter_num % 50 == 0:
        # evaluate both the train and test acc
        model.eval()
        with torch.no_grad():
            train_mean = eval_split(trainer.device, model, train_dataset)
            test_mean  = eval_split(trainer.device, model, test_dataset)
        print(f'the mean of train and test are {train_mean}, {test_mean}')
        # save the model and terminate the training
        if test_mean >= 0.9:
            print(f"reach threshold 0.9 in iteration: {trainer.iter_num}")
            print(f"saving model with test_mean: {test_mean}")
            ckpt_path = os.path.join(f"test/{trainer.config.task}", "model_last.pt")
            torch.save(model.state_dict(), ckpt_path)
            return trainer.iter_num
        # revert model to training mode
        model.train()
    return -1

def plot(result: list, config: CN):
    # histogram
    plt.hist(result, color='blue', alpha=0.7)

    # Add titles and labels
    plt.title(f'{config.trainer.task} - {config.model.init_type} - {config.system.init_seed}')
    plt.xlabel('Iterations')
    plt.ylabel('Frequency')

    # Show the plot
    plt.savefig(f'histogram_{config.model.init_type}_{config.system.init_seed}.png')

def write2txt(result: list, config: CN, fname: str = None):
    # Open the file in write mode ('w')
    if fname is None:
        fname = f'output_{config.model.init_type}_{config.trainer.task}.txt'
    with open(fname, 'w') as file:
        for item in result:
            # Write each item on a new line
            file.write(f"{item}\n")

    print("Data has been written to file.")

# -----------------------------------------------------------------------------

def set_seed_all(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weight_init_test():
    config = get_config()
    setup_logging(config)
    print(config.model.init_type)

    # TODO: try different seed for model
    set_seed(config.system.init_seed)

    # TODO: try different seed to adjust the data order of train/test-set
    data_seed = 0
    train_dataset = ChickenRabbitDataset(config.data, split='train', seed=data_seed)
    test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=data_seed)
    # train_dataset = GCDDataset(config.data, split='train', seed=seed)
    # test_dataset  = GCDDataset(config.data, split='test', seed=seed)

    # set the correct vocab size: 10, block size: chickenrabbit -> 10, gcd -> 6
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()

    # start training
    result = []
    for i in tqdm(range(100)):
        model = GPT(config.model)
        trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
        trainer.set_callback('on_batch_end', batch_end_callback)
        stop_iteration = trainer.run()
        if stop_iteration != -1:
            print(f'The final iteration of this round is {stop_iteration}!')
        else:
            print('It cannot reach 0.9 acc within max_iteration steps...')
        result.append(stop_iteration)
    
    print(result)
    plot(result, config)
    fname = f'output_{config.model.init_type}_sys{config.system.init_seed}_data{data_seed}_{config.trainer.task}.txt'
    write2txt(result, config, fname)

def data_order_test_2_1():
    config = get_config()
    setup_logging(config)
    print(config.model.init_type)

    # start training
    data_seed = 0
    result = []

    if config.trainer.task == "ChickenRabbit":
        train_dataset = ChickenRabbitDataset(config.data, split='train', seed=data_seed)
        test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=data_seed)
    else:
        train_dataset = GCDDataset(config.data, split='train', seed=data_seed)
        test_dataset  = GCDDataset(config.data, split='test', seed=data_seed)

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    
    set_seed(config.system.init_seed)

    # start training
    model = GPT(config.model)

    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    stop_iteration = trainer.run()
    if stop_iteration != -1:
        print(f'The final iteration of this round is {stop_iteration}!')
    else:
        print('It cannot reach 0.9 acc within max_iteration steps...')
    result.append(stop_iteration)

    print(result)

def data_order_test_2_2():
    config = get_config()
    setup_logging(config)
    print(config.model.init_type)

    # start training
    data_seed = 0
    result = []
    
    if config.trainer.task == "ChickenRabbit":
        train_dataset = ChickenRabbitDataset(config.data, split='train', seed=data_seed)
        test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=data_seed)
    else:
        train_dataset = GCDDataset(config.data, split='train', seed=data_seed)
        test_dataset  = GCDDataset(config.data, split='test', seed=data_seed)

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    
    for i in tqdm(range(10)):
        set_seed(i)
        model = GPT(config.model)

        trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
        trainer.set_callback('on_batch_end', batch_end_callback)
        stop_iteration = trainer.run()
        if stop_iteration != -1:
            print(f'The final iteration of this round is {stop_iteration}!')
        else:
            print('It cannot reach 0.9 acc within max_iteration steps...')
        result.append(stop_iteration)

    print(result)

def data_order_test_2_2_gen_sample():
    config = get_config()
    setup_logging(config)
    print(config.model.init_type)

    # start training
    result = []
    
    for i in tqdm(range(10)):
        set_seed(i)
        for j in range(10):
            data_seed = i*10 + j

            if config.trainer.task == "ChickenRabbit":
                train_dataset = ChickenRabbitDataset(config.data, split='train', seed=data_seed)
                test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=data_seed)
            else:
                train_dataset = GCDDataset(config.data, split='train', seed=data_seed)
                test_dataset  = GCDDataset(config.data, split='test', seed=data_seed)

            config.model.vocab_size = train_dataset.get_vocab_size()
            config.model.block_size = train_dataset.get_block_size()

            model = GPT(config.model)

            trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
            trainer.set_callback('on_batch_end', batch_end_callback)
            stop_iteration = trainer.run()
            if stop_iteration != -1:
                print(f'The final iteration of this round is {stop_iteration}!')
            else:
                print('It cannot reach 0.9 acc within max_iteration steps...')
            result.append(stop_iteration)

    print(result)
    write2txt(result, config)

def main():
    # weight_init_test()
    # data_order_test()
    data_order_test_2_1()
    # data_order_test_2_2_gen_sample()

if __name__ == '__main__':
    main()    