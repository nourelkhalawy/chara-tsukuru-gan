# Chara Tsukuru GAN: RPG character generator

Old school JRPG character sprite sheet generator adopted from chainer's example implementation of DCGAN (https://github.com/chainer/chainer).

As default, DCGAN will be trained on input-sprite dataset (~700 32x32 png images), which is created from character sheets collected from indie game developer communities and fanart platform such as DeviantArt.

`python train_dcgan.py --dataset 'input-sprite' --gpu -1`
will turn off the gpu option.

Sample output after 7200 epoch:

![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/example_output_7200.png)

Examples of the features 'extracted':

![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/vec00000029.png)
![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/vec00000054.png)
![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/vec00000063.png)

### Generating characters

* `python train_dcgan.py --dataset 'input-sprite' --print_pan result/snapshot_iter_7200.npz`
will pan through vector spaces from a snapshot `result/snapshot_iter_7200.npz`

![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/pan00000005.png)
![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/pan00000018.png)
![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/pan00000024.png)

* `python train_dcgan.py --dataset 'input-sprite' --print_walk result/snapshot_iter_7200.npz`
will pick two random points and then generate interpolations from a snapshot `result/snapshot_iter_7200.npz`

![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/walk-1-random-random.png)
![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/walk-2-random-random.png)
![example result](https://raw.githubusercontent.com/almchung/chara-tsukuru-gan/master/examples/walk-36-random-random.png)