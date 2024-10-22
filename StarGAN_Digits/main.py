import os
import torch
import random
import datetime
import argparse
import numpy as np
import torch.nn as nn
from solver import Solver
from utils import print_args

def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.data_path,   exist_ok=True)
    os.makedirs(args.model_path,  exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    solver = Solver(args)
    solver.generate_sample_images()         # Generate Sample/GT Images
    solver.train()                          # Training function
    solver.generate_images()                # Generate Images


# Update arguments
def update_args(args):
    args.datasets.sort()

    args.model_path  = os.path.join(args.model_path, "_".join(args.datasets))
    args.output_path = os.path.join(args.output_path, "_".join(args.datasets))

    args.n_domains   = len(args.datasets)

    args.is_cuda = torch.cuda.is_available()
    if args.is_cuda:
        print("Using GPU")
    else:
        print("Cuda not available. Using CPU.")
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StarGAN-digits')

    # Data arguments
    parser.add_argument('--datasets', type=str, nargs='+', default=['mnist', 'cmnist', 'emnist', 'mnistm'], help='dataset to use')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--n_dis_per_gen_update', type=int, default=5, help='number of times critic is updated per 1 generator update')
    parser.add_argument('--gradient_penalty', type=int, default=10, help='gradient penalty hyperparamter')

    # Model arguments
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    # Image generation arguments
    parser.add_argument('--n_images_to_display', type=int, default=50, help='number of images to display per class')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store generated images')

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
