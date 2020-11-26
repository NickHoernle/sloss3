import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import numpy as np

from wideresnet import WideResNet

from symbolic import (
    get_cifar10_experiment_params,
    calc_logic_loss,
    LogicNet
)

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--dataset_path', default='../data', type=str,
                    help='path to where the data are stored')
parser.add_argument("--checkpoint_dir", default="runs")
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--sloss_weight', default=0.1, type=float, help='Weight for the sloss logic term')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--no-sloss', dest='sloss', action='store_false',
                    help='whether to use semantic logic loss (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)
parser.set_defaults(sloss=True)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure(args.checkpoint_dir+"/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](args.dataset_path, train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](args.dataset_path, train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate,
                            semantic_loss=args.sloss, device=device)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    calc_logic = None

    if args.sloss:
        examples, logic_fn = get_cifar10_experiment_params(train_loader.dataset)
        assert logic_fn(torch.arange(10), examples).all()

        examples = examples.to(device)

        logic_net = LogicNet(num_classes=len(train_loader.dataset.classes))
        logic_net.to(device)

        logic_optimizer = torch.optim.Adam(logic_net.parameters(), 1e-2)
        logic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(logic_optimizer, len(train_loader) * args.epochs)

        decoder_optimizer = torch.optim.Adam(model.global_paramters, 1e-2)
        decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, len(train_loader) * args.epochs)

        calc_logic = lambda predictions, targets: calc_logic_loss(predictions, targets, logic_net, logic_fn, device)

        # override the oprimizer from above
        optimizer = torch.optim.SGD(model.local_parameters,
                                    args.lr,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, logic_net,
              criterion, examples,
              optimizer, logic_optimizer, decoder_optimizer,
              scheduler, logic_scheduler, decoder_scheduler,
              epoch, args, calc_logic, device=device)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, args, device=device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train_logic_step(model, logic_net, calc_logic, examples, logic_optimizer, decoder_optimizer, logic_scheduler, decoder_scheduler, device):
    # train the logic net
    logic_net.train()
    model.eval()

    logic_optimizer.zero_grad()

    samps, tgts, thet = model.sample(1000)
    preds, true = calc_logic(samps, tgts)
    logic_loss = F.binary_cross_entropy_with_logits(preds, true.float())
    preds, true = calc_logic(examples, torch.arange(10).to(device))
    logic_loss += F.binary_cross_entropy_with_logits(preds, torch.ones_like(preds))

    logic_loss.backward()
    torch.nn.utils.clip_grad_norm_(logic_net.parameters(), 1.)
    logic_optimizer.step()

    logic_net.eval()
    model.train()
    # train the network to obey the logic
    decoder_optimizer.zero_grad()

    samps, tgts, thet = model.sample(1000)
    preds, true = calc_logic(samps, tgts)
    logic_loss_ = F.binary_cross_entropy_with_logits(preds, torch.ones_like(preds), reduction="none")
    # loss = logic_loss_.mean()
    loss = 0
    loss += logic_loss_[~true].sum() / len(true)
    loss += F.cross_entropy(samps, tgts)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    decoder_optimizer.step()

    logic_scheduler.step()
    decoder_scheduler.step()

    return loss, logic_loss


def train(train_loader, model, logic_net,
          criterion, examples,
          optimizer, logic_optimizer, decoder_optimizer,
          scheduler, logic_scheduler, decoder_scheduler,
          epoch, params, calc_logic=None, device="cuda"):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    net_logic_losses = AverageMeter()
    logic_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)
        target = target.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)

        # compute output
        if not params.sloss:
            output = model(input)
            loss = criterion(output, target)
        else:
            net_logic_loss, logic_loss = train_logic_step(model, logic_net, calc_logic, examples,
                             logic_optimizer, decoder_optimizer, logic_scheduler, decoder_scheduler, device=device)

            output, (mu, lv), theta = model(input)
            recon_loss = criterion(output, target)
            loss = 0
            loss += recon_loss
            kld = -0.5 * (1 + lv - np.log(9.) - (mu.pow(2) + lv.exp()) / 9.).sum(dim=1).mean()
            loss += kld

            preds, true = calc_logic(output, target)
            logic_loss_ = F.binary_cross_entropy_with_logits(preds, torch.ones_like(preds), reduction="none")
            # loss += logic_loss_.mean()
            weight = np.max([1., epoch / 25])
            loss += params.sloss_weight*weight*logic_loss_[~true].sum() / len(true)

            logic_losses.update(logic_loss.data.item(), 1000)
            net_logic_losses.update(net_logic_loss.data.item(), 1000)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Logic Loss {logic_loss.val:.4f} ({logic_loss.avg:.4f})\t'
                  'Net Logic Loss {net_logic_loss.val:.4f} ({net_logic_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, logic_loss=logic_losses, net_logic_loss=net_logic_losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
        log_value('train_logic_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch, params, device="cuda"):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)
        target = target.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)

        # compute output
        if not params.sloss:
            with torch.no_grad():
                output = model(input)
            loss = criterion(output, target)
        else:
            with torch.no_grad():
                output, (mu, lv), theta = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = args.checkpoint_dir+"/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.checkpoint_dir+'/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
