from __future__ import division, absolute_import, print_function
import argparse
from common.util import *

def main(args):
    assert args.dataset in ['mnist', 'cifar' ,'svhn', 'imagenet'], \
        "dataset parameter must be either 'mnist', 'cifar' or 'imagenet'"
    print('Data set: %s' % args.dataset)
    
    if args.dataset == 'mnist':
        from baseline.cnn.cnn_mnist import MNISTCNN as model
        model_mnist = model(mode='train', filename='cnn_{}.pt'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'cifar':
        from baseline.cnn.cnn_cifar10 import CIFAR10CNN as model
        model_cifar = model(mode='train', filename='cnn_{}.pt'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'svhn':
        from baseline.cnn.cnn_svhn import SVHNCNN as model
        model_svhn = model(mode='train', filename='cnn_{}.pt'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset to use; either 'mnist', 'cifar', or 'imagenet'")
    parser.add_argument('-e', '--epochs', required=False, type=int, default=200, help="The number of epochs to train for.")
    parser.add_argument('-b', '--batch_size', required=False, type=int, default=512, help="The batch size to use for training.")
    args = parser.parse_args()
    main(args)
