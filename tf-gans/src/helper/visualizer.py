#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: visualizer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches


class Visualizer(object):
    """ Visualizer the trained model """
    def __init__(self, model, keep_prob=1., save_path=None):
        """ 
        Args:
            model (GANBaseModel): GAN model for visualizing
            keep_prob (float): keep probability for dropout
            save_path (str): directory for saving sampled images.
                Nothing will be saved if None.
        """

        self._save_path = save_path
        self._model = model
        self._latent_op = model.layers['z']
        self._keep_prob = keep_prob

    def viz_2Dlatent_variable(self, sess, dataflow, batch_size=128, file_id=None):
        """
        modify from:
        https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py#L45
        """
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        self._model.set_is_training(False)

        dataflow.setup(epoch_val=0, batch_size=batch_size)
        latent_var_list = []
        label_list = []
        
        while dataflow.epochs_completed == 0:
            batch_data = dataflow.next_batch_dict()
            im = batch_data['im']
            labels = batch_data['label']
            latent_var = sess.run(
                self._latent_op, 
                feed_dict={self._model.encoder_in: im,
                           self._model.keep_prob: self._keep_prob})
            try:           
                latent_var_list.extend(latent_var[:, pick_dim])
            except UnboundLocalError:
                pick_dim = np.random.choice(len(latent_var[0]), 2, replace=False)
                pick_dim = sorted(pick_dim)
                latent_var_list.extend(latent_var[:, pick_dim])

            label_list.extend(labels)

        xs, ys = np.array(latent_var_list).T

        plt.figure()
        kwargs = {'alpha': 0.8}

        classes = set(label_list)
        if classes:
            colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
            kwargs['c'] = [colormap[i] for i in label_list]

            # make room for legend
            ax = plt.subplot(111, aspect='equal')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                        for i, class_ in enumerate(classes)]
            ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                      fancybox=True, loc='center left')

        plt.scatter(xs, ys, s=2, **kwargs)

        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-3.5, 3.5])

        if file_id is not None:
            fig_save_path = os.path.join(self._save_path, 'latent_{}.png'.format(file_id))
        else:
            fig_save_path = os.path.join(self._save_path, 'latent.png')
        plt.savefig(fig_save_path, bbox_inches="tight")
        