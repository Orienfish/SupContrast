from __future__ import print_function

import sys
import argparse
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def plot_tsne(x, y_pred, y_true=None, title='', fig_name=''):
    """
    Plot the TSNE of x, assigned with true labels and pseudo labels respectively.
    Args:
        x: (batch_size, input_dim), raw data to be plotted
        y_pred: (batch_size), optional, pseudo labels for x
        y_true: (batch_size), ground-truth labels for x
        title: str, title for the plots
        fig_name: str, the file name to save the plot
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    tsne = TSNE(2, perplexity=50)
    x_emb = tsne.fit_transform(x)

    if y_true is not None: # Two subplots
        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_pred,
                        palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full", ax=ax1)
        ax1.set_title('Clusters with pseudo labels, {}'.format(title))
        ax2 = plt.subplot(122)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_true,
                        palette=sns.color_palette("hls", np.unique(y_true).size),
                        legend="full", ax=ax2)
        ax2.set_title(title)
    else: # Only one plot for predicted labels
        fig = plt.figure(figsize=(6, 5))
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1],
                        hue=y_pred, palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full")
        plt.title(title)

    if fig_name != '':
        plt.savefig(fig_name, bbox_inches='tight')

    plt.close(fig)

def knn_eval(test_embeddings, test_labels, knn_train_embeddings,
             knn_train_labels, opt):
    """KNN classification and plot in evaluations"""
    # perform kNN classification
    from sklearn.neighbors import KNeighborsClassifier
    st = time.time()
    neigh = KNeighborsClassifier(n_neighbors=50)
    pred_labels = neigh.fit(knn_train_embeddings, knn_train_labels).predict(test_embeddings)
    knn_time = time.time() - st
    knn_acc = np.sum(pred_labels == test_labels) / pred_labels.size

    print('ckpt: {} knn_acc: {}'.format(opt.ckpt, knn_acc))

    # plot t-SNE for test embeddings
    model_name = opt.ckpt.split('/')[-1].split('.')[0]
    print(model_name)
    plot_tsne(test_embeddings, pred_labels, test_labels,
              title='knn acc: {}'.format(knn_acc),
              fig_name='tsne_{}_{}.png'.format(opt.dataset, model_name))



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion



def validate(train_loader, val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    test_labels, knn_labels = [], []
    test_embeddings, knn_embeddings = None, None

    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            embeddings = model.encoder(images).detach().cpu().numpy()
            if knn_embeddings is None:
                knn_embeddings = embeddings
            else:
                knn_embeddings = np.concatenate((knn_embeddings, embeddings), axis=0)
            knn_labels += labels.detach().tolist()
        knn_labels = np.array(knn_labels).astype(int)

        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # collect embeddings
            embeddings = model.encoder(images).detach().cpu().numpy()
            if test_embeddings is None:
                test_embeddings = embeddings
            else:
                test_embeddings = np.concatenate((test_embeddings, embeddings), axis=0)
            test_labels += labels.detach().tolist()


        test_labels = np.array(test_labels).astype(int)

        knn_eval(test_embeddings, test_labels, knn_embeddings, knn_labels, opt)



def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    #for epoch in range(1, opt.epochs + 1):
    #    adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
    #    time1 = time.time()
    #    loss, acc = train(train_loader, model, classifier, criterion,
    #                      optimizer, epoch, opt)
    #    time2 = time.time()
    #    print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
    #        epoch, time2 - time1, acc))

    #    # eval for one epoch
    #    loss, val_acc = validate(val_loader, model, classifier, criterion, opt,
    #                             epoch)
    #    if val_acc > best_acc:
    #        best_acc = val_acc

    #print('best accuracy: {:.2f}'.format(best_acc))

    validate(train_loader, val_loader, model, classifier, criterion, opt)

if __name__ == '__main__':
    main()
