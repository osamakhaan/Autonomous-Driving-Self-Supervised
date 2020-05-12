'''
This file is used to create the permutations list for pretraining with the jigsaw and stereo pretext tasks.
Permutations are selected based on maximum hamming distance.
'''

# borrowed from: https://github.com/bbrattoli/JigsawPuzzlePytorch

import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--classes', required=True, type=int,
                    help='Number of permutations to select')
parser.add_argument('--type', required=True, type=str,
                    help="Select 'jigsaw' or 'stereo'")
parser.add_argument('--selection', default='max', type=str,
                    help='Sample selected per iteration based on hamming distance: [max] highest; [mean] average')
args = parser.parse_args()

if __name__ == "__main__":
    if args.type == "jigsaw":
        limit = 9
    elif args.type == "stereo":
        limit = 6
    else:
        raise RuntimeError("Invalid selection for type. Select 'jigsaw' or 'stereo'")


    outname = f'permutations_{args.type}_{args.classes}'

    P_hat = np.array(list(itertools.permutations(list(range(limit)), limit)))
    n = P_hat.shape[0]

    for i in trange(args.classes):
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if args.selection == 'max':
            j = D.argmax()
        else:
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]

        if i % 100 == 0:
            np.save(outname, P)

    np.save(outname, P)
    print('file created --> ' + outname)
