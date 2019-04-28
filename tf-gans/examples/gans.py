#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gans.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('/home/nowgeun1/Desktop/infogan/tf-gans/')


from src.nets.infogan import infoGAN
from src.helper.trainer import Trainer
from src.helper.generator import Generator
import loader as loader
# from src.helper.visualizer import Visualizer
# import src.models.distribution as distribution


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--generate', action='store_true',
                        help='Sampling from trained model')
    parser.add_argument('--test', action='store_true',
                        help='test')

    parser.add_argument('--gan_type', type=str, default='infogan',
                        help='Type of GAN for experiment.')
    parser.add_argument('--dataset', type=str, default='celeba',
                        help='Dataset used for experiment.')

    parser.add_argument('--zlen', type=int, default=100,
                        help='length of random vector z')

    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Init learning rate')
    parser.add_argument('--keep_prob', type=float, default=1.,
                        help='keep_prob')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=50,
                        help='Max iteration')

    parser.add_argument('--ng', type=int, default=1,
                        help='number generator training each step')
    parser.add_argument('--nd', type=int, default=1,
                        help='number discriminator training each step')

    parser.add_argument('--w_mutual', type=float, default=1.0,
                        help='weight of mutual information loss for InfoGAN')
    parser_add_argument('--dir', type=str,
                        help='your working directory, for example /Users/nowgeun/Desktop/infogan')

    return parser.parse_args()

if FLAGS.dir.endswith("/"):
    wd = FLAGS.dir[:-1]
else:
    wd = FLAGS.dir


SAVE_PATH = '{}/data/out/gans/'.format(wd)
MNIST_PATH = '{}/data/MNIST_data/'.format(wd)
CELEBA_PATH = '{}/data/celebA/'.format(wd)


def train():
    if FLAGS.gan_type == 'infogan':
        train_info()
    else:
        raise ValueError('Wrong GAN type!')

def train_info():
    FLAGS = get_args()
    if FLAGS.gan_type == 'infogan':
        gan_model = infoGAN
        print('**** InfoGAN ****')
    else:
        raise ValueError('Wrong GAN type!')

    save_path = os.path.join(SAVE_PATH, FLAGS.gan_type)
    save_path += '/'

    # load dataset
    if FLAGS.dataset == 'celeba':
        im_size = 32
        n_channels = 3
        n_continuous = 5
        n_discrete = 0
        cat_n_class_list = [10 for i in range(n_discrete)]
        max_grad_norm = 0.
       
        train_data = loader.load_celeba(
            FLAGS.bsize, data_path=CELEBA_PATH, rescale_size=im_size)
    else:
        im_size = 28
        n_channels = 1
        n_continuous = 4
        n_discrete = 1
        cat_n_class_list = [10]
        max_grad_norm = 10.

        train_data = loader.load_mnist(FLAGS.bsize, data_path=MNIST_PATH)
        
    train_model = gan_model(
        input_len=FLAGS.zlen, im_size=im_size, n_channels=n_channels,
        cat_n_class_list=cat_n_class_list,
        n_continuous=n_continuous, n_discrete=n_discrete,
        mutual_info_weight=FLAGS.w_mutual, max_grad_norm=max_grad_norm)
    train_model.create_train_model()

    generate_model = gan_model(
        input_len=FLAGS.zlen, im_size=im_size, n_channels=n_channels,
        cat_n_class_list=cat_n_class_list,
        n_continuous=n_continuous, n_discrete=n_discrete)
    generate_model.create_generate_model()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(save_path)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(FLAGS.maxepoch):
            train_model.train_epoch(
                sess, train_data, init_lr=FLAGS.lr,
                n_g_train=FLAGS.ng, n_d_train=FLAGS.nd, keep_prob=FLAGS.keep_prob,
                summary_writer=writer)
            generate_model.generate_samples(
                sess, keep_prob=FLAGS.keep_prob, file_id=epoch_id, save_path=save_path)
            saver.save(sess,'{}gan-{}-epoch-{}'.format(save_path+ FLAGS.dataset, FLAGS.gan_type, epoch_id))
        saver.save(sess, '{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))



if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()



