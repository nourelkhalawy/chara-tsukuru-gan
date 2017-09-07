#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from net import Discriminator
from net import Generator
from updater import DCGANUpdater
from visualize import out_generated_image
from visualize import EIGEN_generated_image
from visualize import RANDOM_generated_image
from visualize import PANNING_generated_image
from visualize import WALKING_generated_image


def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.  Default is input-sprite.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=200,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=50,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--print', default='',
                        help='Generate images only; enter snapshot name')
    parser.add_argument('--print_pan', default='',
                        help='Generate images only; enter snapshot name')
    parser.add_argument('--print_walk', default='',
                        help='Generate images only; enter snapshot name')
    parser.add_argument('--walk_start', type=str, default='null',
                        help='secondary parameters for print_walk; starting position str array, [-1,1]')
    parser.add_argument('--walk_end', type=str, default='null',
                        help='secondary parameters for print_walk; end position str array, [-1,1]')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    #def make_optimizer(model, alpha=0.0002, beta1=0.5):
    #    optimizer = chainer.optimizers.Adam(alpha, beta1)
    #def make_optimizer(model, rho=0.95, eps=1e-06):
    #    optimizer = chainer.optimizers.AdaDelta(rho, eps)
    #def make_optimizer(model, lr=0.00001, eps=1e-08):
    #    optimizer = chainer.optimizers.AdaGrad(lr, eps)

    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha, beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if args.dataset == '':
        # Load the CIFAR10 dataset if args.dataset is not specified
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'
              .format(args.dataset, len(image_files)))
        train = chainer.datasets\
            .ImageDataset(paths=image_files, root=args.dataset)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    if args.print:
        # Load the snapshot
        chainer.serializers.load_npz(args.print, trainer)
        print(args.print)

        RANDOM_generated_image(
            gen, dis,
            20, 20, args.seed, args.out, trainer.updater.iteration)

        for idx in range(0,args.n_hidden,1):
            EIGEN_generated_image(
                gen, dis,
                8, 8, args.seed, args.out, idx)

    if args.print_pan:
        # Load the snapshot
        chainer.serializers.load_npz(args.print_pan, trainer)
        print(args.print_pan)

        RANDOM_generated_image(
            gen, dis,
            20, 20, args.seed, args.out, trainer.updater.iteration)

        for idx in range(0,40,1):
            PANNING_generated_image(
                gen, dis,
                8, 8, args.seed, args.out, idx)

    if args.print_walk:
        # Load the snapshot
        chainer.serializers.load_npz(args.print_walk, trainer)
        print(args.print_walk)

        RANDOM_generated_image(
            gen, dis,
            20, 20, args.seed, args.out, trainer.updater.iteration)

        WALKING_generated_image(
            gen, dis,
            8, 8, args.seed, args.out, args.n_hidden, args.walk_start, args.walk_end)

    else:
        # Run the training
        trainer.run()


if __name__ == '__main__':
    main()
