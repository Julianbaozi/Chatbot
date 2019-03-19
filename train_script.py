from train import *
import argparse

import os
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--teaching', type=str2bool, default='False')
parser.add_argument('--attn', type=str2bool, default='True')
args = parser.parse_args()
opt = vars(args)
if __name__ == "__main__":
    config['num_layers'] = opt['num_layers']
    config['teaching'] = opt['teaching']
    config['attn'] =opt['attn']
    train_loss,val_loss = train()
