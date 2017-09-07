#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image

def EIGEN_generated_image(gen, dis, rows, cols, seed, dst, idx):
    np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp

    print(" [*] %d" % idx)

    #z = Variable(xp.asarray(gen.walk_hidden(n_images, idx)))
    z = Variable(xp.asarray(gen.show_hidden(n_images, idx)))

    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    preview_dir = '{}/preview'.format(dst)
    preview_path = preview_dir +\
        '/vec{:0>8}.png'.format(idx)
    print(preview_path)

    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)

def RANDOM_generated_image(gen, dis, rows, cols, seed, dst, itr):
    np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp

    print(" [*] sample output")

    z = Variable(xp.asarray(gen.make_hidden(n_images)))
    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    preview_dir = '{}/preview'.format(dst)
    preview_path = preview_dir +\
        '/output{:0>8}.png'.format(itr)
    print(preview_path)

    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)

def PANNING_generated_image(gen, dis, rows, cols, seed, dst, idx):
    np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp

    print(" [*] %d" % idx)

    z = Variable(xp.asarray(gen.pan_hidden(n_images, idx)))

    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    preview_dir = '{}/preview'.format(dst)
    preview_path = preview_dir +\
        '/pan{:0>8}.png'.format(idx)
    print(preview_path)

    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)

def WALKING_generated_image(gen, dis, rows, cols, seed, dst, n_hidden, start, end):
    np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp

    # check and assign a starting point
    if start == 'null':
        startPoint = np.random.uniform(-1,1,n_hidden).astype(np.float32)
        start = 'random'
    else:
        startPoint = start.split(',')

    # check and assign an end point
    if end == 'null':
        endPoint = np.random.uniform(-1,1,n_hidden).astype(np.float32)
        end = 'random'
    else:
        endPoint = end.split(',')

    print("walk [*]" + start + " [*] %d" + end)

    z = Variable(xp.asarray(gen.walk_hidden(n_images, startPoint, endPoint)))

    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    preview_dir = '{}/preview'.format(dst)
    preview_path = preview_dir +\
        '/walk-' + str(seed) + '-' + start + '-' + end + '.png'
    print(preview_path)

    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)
