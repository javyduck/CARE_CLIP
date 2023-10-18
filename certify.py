# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
import pandas 
from architectures import ARCHITECTURES, robust_clip, get_gcn, load_gcn_from_ckpt
import pandas as pd
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, default = 'cifar10', help="which dataset")
parser.add_argument('--arch', default='ViT-L-14', type=str, choices=ARCHITECTURES,
                    help='the arch for clip')
parser.add_argument("--path", type=str, default='logs/vanilla', help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, default=0.25, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument('--vanilla', default=False, action='store_true', help='use only the clip')
parser.add_argument('--carlini', default=False, action='store_true', help='carlini')
parser.add_argument('--classifier', default=False, action='store_true', help='carlini')
parser.add_argument("--skip", type=int, default=20, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    if args.vanilla:
        outfile = os.path.join('logs/vanilla', f'{args.dataset}_{args.arch}_certification_noise_sd{args.sigma}.txt')
        base_classifier = robust_clip(args.arch,
                              args.dataset, 
                              reasoning = False, 
                              noise_sd = args.sigma, 
                              denoising = True, 
                              gcn_model = None,)
    elif args.carlini:
        outfile = os.path.join('logs/carlini', f'{args.dataset}_certification_noise_sd{args.sigma}.txt')
        base_classifier = robust_clip(args.arch,
                              args.dataset, 
                              reasoning = False, 
                              noise_sd = args.sigma, 
                              denoising = True, 
                              gcn_model = None,
                              use_classifier = True)
    else:
        checkpoint = torch.load(os.path.join(args.path, 'checkpoint.pth.tar'))
        # Combine the directory with the new filename
        outfile = os.path.join(args.path, f'certification_noise_sd{args.sigma}.txt')
        gcn_model = load_gcn_from_ckpt(checkpoint)
        gcn_model.eval()
        base_classifier = robust_clip(checkpoint['clip_arch'],
                                      checkpoint['dataset'], 
                                      reasoning = True, 
                                      knowledge_path = checkpoint['knowledge_path'],
                                      noise_sd = args.sigma, 
                                      denoising = True, 
                                      gcn_model = gcn_model,
                                      use_classifier = checkpoint['classifier'])
    
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # check if output file already exists
    if os.path.exists(outfile):
        df = pd.read_csv(outfile, sep="\t")
        max_idx = df['idx'].max()  # get the maximum index in the existing outfile
    else:
        max_idx = -1  # if file does not exist, set maximum index to -1

    # open the outfile in append mode if it exists, write mode if it doesn't
    f = open(outfile, 'a' if os.path.exists(outfile) else 'w+')

    # print header only if file is empty
    if max_idx == -1:
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # make sure to start from the next index after max_idx
    for i in range(max_idx+1, len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed))

    f.close()