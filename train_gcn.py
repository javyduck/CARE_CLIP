# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
import setGPU
import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, robust_clip, get_gcn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from torchvision.transforms import Resize
from torch_geometric.data import Data
from train_utils import AverageMeter, accuracy, init_logfile, log
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS)
parser.add_argument('--arch', default='ViT-L/14', type=str, choices=ARCHITECTURES,
                    help='the arch for clip')
parser.add_argument('--outdir', default='logs', type=str, help='folder to save model and training log)')
parser.add_argument('--knowledge_path', default='cifar10_predicate_knowledge_top10', type=str, help='the json storing the knowledge')
parser.add_argument('--suffix', default='', type=str, help='file suffix')
parser.add_argument('--eta', default=5.0, type=float,
                    help='the weight for the classification loss')
parser.add_argument('--train_main', default=False, action='store_true',
                    help='train the weight for the main predicate or not')
parser.add_argument('--attention', default=False, action='store_true',
                    help='using attention')
parser.add_argument('--classifier', default=False, action='store_true',
                    help='using sota classifier')
parser.add_argument('--mode', default='sample', type=str, choices=['sample', 'approx'],
                    help='the mode for M-step')
parser.add_argument('--pesudo', default=False, action='store_true',
                    help='pesudo_training')
parser.add_argument('--resume', default=False, action='store_true',
                    help='continue training')
parser.add_argument('--percent', default=1.0, type=float,
                    help='how many percent of the input is true')
parser.add_argument('--sample_num', default=50, type=int, metavar='N',
                    help='the number of samples for EM-step')
parser.add_argument('--embedding_dim', default=512, type=int, metavar='N',
                    help='the embedding dim for predicates')
parser.add_argument('--hidden_dim', default=512, type=int, metavar='N',
                    help='the hidden dim of GCN')
parser.add_argument('--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=500, type=int, metavar='N',
                    help='batchsize (default: 500)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

def main():
    if args.pesudo:
        outdir = os.path.join(args.outdir, f'{args.dataset}/simulated_training/{args.knowledge_path}/{args.arch.replace("/","-")}/train_main_{args.train_main}_attention_{args.attention}_mode_{args.mode}_eta_{args.eta}_embedding_dim_{args.embedding_dim}_hidden_dim_{args.hidden_dim}_noise_sd_{args.noise_sd}_percent_{args.percent}_classifier_{args.classifier}{args.suffix}')
    else:
        outdir = os.path.join(args.outdir, f'{args.dataset}/real_training/{args.knowledge_path}/{args.arch.replace("/","-")}/train_main_{args.train_main}_attention_{args.attention}_mode_{args.mode}_eta_{args.eta}_embedding_dim_{args.embedding_dim}_hidden_dim_{args.hidden_dim}_noise_sd_{args.noise_sd}_classifier_{args.classifier}{args.suffix}')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)
    
    clip_model = robust_clip(args.arch, args.dataset, reasoning = True, knowledge_path = args.knowledge_path, use_classifier = args.classifier)
    clip_model.eval()
    
    gcn_model = get_gcn(args.dataset, args.knowledge_path, args.eta, args.sample_num, args.embedding_dim, args.hidden_dim, args.train_main, args.attention, args.mode)
    
    logfilename = os.path.join(outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    # separate out the parameters
#     w_params = [params for name, params in gcn_model.named_parameters() if name in ['w', 'edge_weight']]
#     base_params = [params for name, params in gcn_model.named_parameters() if name not in ['w', 'edge_weight']]

#     # create a list of parameter groups
#     param_groups = [{'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}, 
#                    {'params': w_params, 'lr': args.lr * 100, 'weight_decay': args.weight_decay}]

#     # pass these parameter groups to the optimizer
#     optimizer = Adam(param_groups)

    optimizer = Adam(gcn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
    if args.resume:
        checkpoint = torch.load(os.path.join(outdir, 'checkpoint.pth.tar'))
        begin = checkpoint['epoch']
        gcn_model.load_state_dict(checkpoint['gcn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        begin = 0
        
    best_test_acc = 0.0
    for epoch in range(begin, args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, args.pesudo, args.percent, clip_model, gcn_model, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, clip_model, gcn_model, optimizer, epoch, args.noise_sd)
        after = time.time()
        scheduler.step()
        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_last_lr()[0], train_loss, train_acc, test_loss, test_acc))
        # Only save the model if the current test accuracy is greater than the best one seen so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'dataset': args.dataset,
                'knowledge_path': args.knowledge_path,
                'train_main': args.train_main,
                'attention': args.attention,
                'mode': args.mode,
                'eta': args.eta,
                'sample_num': args.sample_num,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'clip_arch': args.arch,
                'classifier': args.classifier,
                'gcn_state_dict': gcn_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(outdir, 'checkpoint.pth.tar'))
        
        torch.save({
                'epoch': epoch + 1,
                'dataset': args.dataset,
                'knowledge_path': args.knowledge_path,
                'train_main': args.train_main,
                'attention': args.attention,
                'mode': args.mode,
                'eta': args.eta,
                'sample_num': args.sample_num,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'clip_arch': args.arch,
                'classifier': args.classifier,
                'gcn_state_dict': gcn_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(outdir, 'last_checkpoint.pth.tar'))

def train(loader: DataLoader, pesudo: bool, percent: float, clip_model: torch.nn.Module, gcn_model: torch.nn.Module, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    gcn_model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if pesudo:
            confidence, targets = gcn_model.get_simulated_input(inputs.shape[0], percent)
            confidence = confidence.cuda()
            targets = targets.cuda()
        else:
            inputs = inputs.cuda()
            targets = targets.cuda()
            # augment inputs with noise
            inputs += torch.randn_like(inputs, device='cuda') * noise_sd
            # compute output
            confidence = clip_model(inputs, only_main = False)
        outputs, loss = gcn_model.E_step(confidence, targets)
        # measure accuracy and record loss
        main_num = gcn_model.main_num
        acc1, acc5 = accuracy(outputs[:, :main_num], targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        gcn_model.w.grad = -gcn_model.M_step(confidence)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, clip_model: torch.nn.Module, gcn_model: torch.nn.Module, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    gcn_model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            confidence = clip_model(inputs, only_main = False)
            # compute output
            outputs, loss = gcn_model.E_step(confidence, targets)
            
            # measure accuracy and record loss
            main_num = gcn_model.main_num
            acc1, acc5 = accuracy(outputs[:, :main_num], targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)
    
if __name__ == "__main__":
    main()