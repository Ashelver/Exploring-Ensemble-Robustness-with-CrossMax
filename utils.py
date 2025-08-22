# a hack to ensure scripts search cwd
import sys
sys.path.append('.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import argparse
from convex_adversarial import robust_loss, robust_loss_parallel
from convex_adversarial import epsilon_from_model, DualNetBounds
from convex_adversarial import Dense, DenseSequential
import math
import os
import time
from collections import Counter


DEBUG = False

mean = [0.485, 0.456, 0.406]
std = [0.225, 0.225, 0.225]

class NormalizedModel(nn.Module):
    def __init__(self, model, mean = mean, std=std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor(std).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


class AverageLogitModel(nn.Module):
    def __init__(self, models):
        super(AverageLogitModel, self).__init__()
        self.models = models  # list of NormalizedModel wrapped sub-models

    def forward(self, x):
        logits = [model(x) for model in self.models]
        return sum(logits) / len(logits)


def model_wide(in_ch, out_width, k): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*k, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*k*out_width*out_width,k*128),
        nn.ReLU(),
        nn.Linear(k*128, 10)
    )
    return model

def model_deep(in_ch, out_width, k, n1=8, n2=16, linear_size=100): 
    def group(inf, outf, N): 
        if N == 1: 
            conv = [nn.Conv2d(inf, outf, 4, stride=2, padding=1), 
                         nn.ReLU()]
        else: 
            conv = [nn.Conv2d(inf, outf, 3, stride=1, padding=1), 
                         nn.ReLU()]
            for _ in range(1,N-1):
                conv.append(nn.Conv2d(outf, outf, 3, stride=1, padding=1))
                conv.append(nn.ReLU())
            conv.append(nn.Conv2d(outf, outf, 4, stride=2, padding=1))
            conv.append(nn.ReLU())
        return conv

    conv1 = group(in_ch, n1, k)
    conv2 = group(n1, n2, k)


    model = nn.Sequential(
        *conv1, 
        *conv2,
        Flatten(),
        nn.Linear(n2*out_width*out_width,linear_size),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size, shuffle_test=False): 
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def fashion_mnist_loaders(batch_size): 
    mnist_train = datasets.MNIST("./fashion_mnist", train=True,
       download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./fashion_mnist", train=False,
       download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def mnist_500(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,500),
        nn.ReLU(),
        nn.Linear(500, 10)
    )
    return model


def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_model_wide(k): 
    return model_wide(1, 7, k)

def mnist_model_deep(k): 
    return model_deep(1, 7, k)

def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

def replace_10_with_0(y): 
    return y % 10

def svhn_loaders(batch_size): 
    train = datasets.SVHN("./data", split='train', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    test = datasets.SVHN("./data", split='test', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def svhn_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).cuda()
    return model

def har_loaders(batch_size):     
    X_te = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/test/X_test.txt')).float()
    X_tr = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/train/X_train.txt')).float()
    y_te = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/test/y_test.txt')-1).long()
    y_tr = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/train/y_train.txt')-1).long()

    har_train = td.TensorDataset(X_tr, y_tr)
    har_test = td.TensorDataset(X_te, y_te)

    train_loader = torch.utils.data.DataLoader(har_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(har_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def har_500_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 6)
    )
    return model

def har_500_250_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 6)
    )
    return model

def har_500_250_100_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 100),
        nn.ReLU(),
        nn.Linear(100, 6)
    )
    return model

def har_resnet_model(): 
    model = DenseSequential(
        Dense(nn.Linear(561, 561)), 
        nn.ReLU(), 
        Dense(nn.Sequential(), None, nn.Linear(561,561)),
        nn.ReLU(), 
        nn.Linear(561,6)
        )
    return model

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # test = datasets.CIFAR10('./data', train=False, 
    #     transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def cifar_loaders_no_normalizred(batch_size, shuffle_test=False): 
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_resnet(N = 5, factor=10): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, False)
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*8*8,1000), 
        nn.ReLU(), 
        nn.Linear(1000, 10)]
        )
    model = DenseSequential(
        *layers
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model

def argparser(batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-3, 
              epsilon=0.1, starting_epsilon=None, 
              proj=None, 
              norm_train='l1', norm_test='l1', 
              opt='sgd', momentum=0.9, weight_decay=5e-4): 

    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=10)

    # projection settings
    parser.add_argument('--proj', type=int, default=proj)
    parser.add_argument('--norm_train', default=norm_train)
    parser.add_argument('--norm_test', default=norm_test)

    # model arguments
    parser.add_argument('--model', default=None)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', default=None)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)


    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    
    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon 
    if args.prefix: 
        if args.model is not None: 
            args.prefix += '_'+args.model

        if args.method is not None: 
            args.prefix += '_'+args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval', 
                  'method', 'model', 'cuda_ids', 'load', 'real_time', 
                  'test_batch_size']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # Ignore these parameters for filename since we never change them
        banned += ['momentum', 'weight_decay']

        if args.cascade == 1: 
            banned += ['cascade']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        # if args.model != 'resnet': 
        banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))

        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids


    return args

def args2kwargs(args, X=None): 

    if args.proj is not None: 
        kwargs = {
            'proj' : args.proj, 
        }
    else:
        kwargs = {
        }
    kwargs['parallel'] = (args.cuda_ids is not None)
    return kwargs



def argparser_evaluate(epsilon=0.1, norm='l1'): 

    parser = argparse.ArgumentParser()

    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument('--proj', type=int, default=None)
    parser.add_argument('--norm', default=norm)
    parser.add_argument('--model', default=None)
    parser.add_argument('--dataset', default='mnist')

    parser.add_argument('--load')
    parser.add_argument('--output')

    parser.add_argument('--real_time', action='store_true')
    # parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=True)
    parser.add_argument('--cuda_ids', default=None)

    
    args = parser.parse_args()

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids


    return args

def robust_loss_cascade(models, epsilon, X, y, **kwargs): 
    total_robust_ce = 0.
    total_ce = 0.
    total_robust_err = 0.
    total_err = 0.

    batch_size = float(X.size(0))

    I = torch.arange(X.size(0)).type_as(y.data)

    if X.size(0) == 1: 
        rl = robust_loss_parallel
    else:
        rl = robust_loss

    for j,model in enumerate(models[:-1]): 

        out = model(X)
        ce = nn.CrossEntropyLoss(reduce=False)(out, y)

        _, uncertified = rl(model, epsilon, X,
                                     out.max(1)[1],
                                     size_average=False, **kwargs)

        certified = ~uncertified
        l = []
        if certified.sum() == 0: 
            pass
            # print("Warning: Cascade stage {} has no certified values.".format(j+1))
        else: 
            X_cert = X[Variable(certified.nonzero()[:,0])]
            y_cert = y[Variable(certified.nonzero()[:,0])]

            ce = ce[Variable(certified.nonzero()[:,0])]
            out = out[Variable(certified.nonzero()[:,0])]
            err = (out.data.max(1)[1] != y_cert.data).float()
            robust_ce, robust_err = rl(model, epsilon, 
                                                 X_cert, 
                                                 y_cert, 
                                                 size_average=False,
                                                 **kwargs)
            # add statistics for certified examples
            total_robust_ce += robust_ce.sum()
            total_ce += ce.data.sum()
            total_robust_err += robust_err.sum()
            total_err += err.sum()
            l.append(certified.sum())
            # reduce data set to uncertified examples
            if uncertified.sum() > 0: 
                X = X[Variable(uncertified.nonzero()[:,0])]
                y = y[Variable(uncertified.nonzero()[:,0])]
                I = I[uncertified.nonzero()[:,0]]
            else: 
                robust_ce = total_robust_ce/batch_size
                ce = total_ce/batch_size
                robust_err = total_robust_err.item()/batch_size
                err = total_err.item()/batch_size
                return robust_ce, robust_err, ce, err, None
        ####################################################################
    # compute normal ce and robust ce for the last model
    out = models[-1](X)
    ce = nn.CrossEntropyLoss(reduce=False)(out, y)
    err = (out.data.max(1)[1] != y.data).float()

    robust_ce, robust_err = rl(models[-1], epsilon, X, y,
                                         size_average=False, **kwargs)

    # update statistics with the remaining model and take the average 
    total_robust_ce += robust_ce.sum()
    total_ce += ce.data.sum()
    total_robust_err += robust_err.sum()
    total_err += err.sum()

    robust_ce = total_robust_ce/batch_size
    ce = total_ce/batch_size
    robust_err = total_robust_err.item()/batch_size
    err = total_err.item()/batch_size

    _, uncertified = rl(models[-1], epsilon, 
                                 X, 
                                 out.max(1)[1], 
                                 size_average=False,
                                 **kwargs)
    if uncertified.sum() > 0: 
        I = I[uncertified.nonzero()[:,0]]
    else:
        I = None

    return robust_ce, robust_err, ce, err, I


def evaluate_robust_cascade(loader, models, epsilon, epoch, log, verbose, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    for model in models:
        model.eval()

    torch.set_grad_enabled(False)
    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        robust_ce, robust_err, ce, err, _ = robust_loss_cascade(models, 
                                                             epsilon, 
                                                             Variable(X), 
                                                             Variable(y), 
                                                             **kwargs)

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.item(), robust_err, ce.item(), err,
           file=log)
        if verbose: 
            endline = '\n' if  i % verbose == 0 else '\r'
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        log.flush()

        del X, y, robust_ce, ce
        if DEBUG and i == 10: 
            break
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    torch.set_grad_enabled(True)
    return robust_errors.avg


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



def select_cifar_model(m):
    m = m.lower()  # 忽略大小写
    if 'large' in m:
        return cifar_model_large().cuda()
    elif 'resnet' in m:
        return cifar_model_resnet(N=1, factor=1).cuda()
    elif 'small' in m:
        return cifar_model().cuda()
    else:
        raise ValueError(f"Unknown model type in select_model: {m}")



def load_lp_models():
    MODEL_CONFIGS = {
    'resnet_8px': 'models_scaled/cifar_resnet_8px.pth',
    'resnet_2px': 'models_scaled/cifar_resnet_2px.pth',
    'large_8px':  'models_scaled/cifar_large_8px.pth',
    'large_2px':  'models_scaled/cifar_large_2px.pth',
    'small_8px':  'models_scaled/cifar_small_8px.pth',
    'small_2px':  'models_scaled/cifar_small_2px.pth',
    }

    models = {}
    for name, path in MODEL_CONFIGS.items():
        ckpt = torch.load(path)
        model = select_cifar_model(name)
        if isinstance(ckpt['state_dict'], list):
            model.load_state_dict(ckpt['state_dict'][0])
        else:
            model.load_state_dict(ckpt['state_dict'])
        model.eval()
        models[name] = NormalizedModel(model)
    
    return models


def load_std_models():
    MODEL_CONFIGS = {
        'resnet_1': 'models_standard/cifar_resnet_std_epoch10.pth',
        'resnet_2': 'models_standard/cifar_resnet_std_epoch20.pth',
        'large_1':  'models_standard/cifar_large_std_epoch10.pth',
        'large_2':  'models_standard/cifar_large_std_epoch20.pth',
        'small_1':  'models_standard/cifar_small_std_epoch10.pth',
        'small_2':  'models_standard/cifar_small_std_epoch20.pth',
    }

    models = {}
    for name, path in MODEL_CONFIGS.items():
        ckpt = torch.load(path)
        model = select_cifar_model(name)
        model.load_state_dict(ckpt)
        model.eval()
        models[name] = NormalizedModel(model)

    return models