import os
import torch
import argparse
import datetime
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
    args.data_path   = os.path.join(args.data_path, args.dataset)
    args.model_path  = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    
    args.is_cuda = torch.cuda.is_available()
    if args.is_cuda:
        print("Using GPU")
    else:
        print("Cuda not available. Using CPU.")
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGAN-32x32')

    # Data arguments
    parser.add_argument('--dataset', type=str.lower, default='mnist', choices=['mnist', 'fashionmnist', 'usps'], help='dataset to use')
    parser.add_argument('--image_size', type=int, default=28, help='image size')
    parser.add_argument("--n_channels", type=int, default=1, help='number of channels')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--z_dim', type=int, default=5, help='sample noise dimensions')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')

    # Model arguments
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    # Image generation arguments
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
