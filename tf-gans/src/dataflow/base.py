#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np 
import src.utils.utils as utils
from src.utils.dataflow import get_rng, get_file_list


# def get_file_list(file_dir, file_ext, sub_name=None):
#     # assert file_ext in ['.mat', '.png', '.jpg', '.jpeg']
#     re_list = []

#     if sub_name is None:
#         return np.array([os.path.join(root, name)
#             for root, dirs, files in os.walk(file_dir) 
#             for name in sorted(files) if name.endswith(file_ext)])
#     else:
#         return np.array([os.path.join(root, name)
#             for root, dirs, files in os.walk(file_dir) 
#             for name in sorted(files) if name.endswith(file_ext) and sub_name in name])

class DataFlow(object):
    """ Base class for dataflow 

        Data are read by channels. For example, for image classification,
        the dataflow can be two channels of image channel and label channel.

        To access the data of mini-batch, first get data of all the channels
        through batch_data = DataFlow.next_batch_dict()
        then use corresponding key to get data of a single channel through
        batch_data[key].
    """
    def __init__(self,
                 data_name_list,
                 data_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 load_fnc_list=None,
                 ):
        """
        Args:
            data_name_list (list of str): list of filenames or part of filenames
                of each data channel
            data_dir (list of str): list of directories of each data channel
            shuffle (bool): whether shuffle data or not
            batch_dict_name (list of str): list of keys for each channel of batch data
            load_fnc_list (list): list of pre-process functions for each channel of data
        """
        data_name_list = utils.make_list(data_name_list)
        load_fnc_list = utils.make_list(load_fnc_list)
        data_dir = utils.make_list(data_dir)
        # pad data_dir with the same path only when data_dir is a single input
        if len(data_dir) == 1:
            data_dir_list = [data_dir[0] for i in range(len(load_fnc_list))]
            data_dir = data_dir_list

        dataflow_list = []
        self._load_fnc_list = []
        for data_name, load_fnc in zip(data_name_list, load_fnc_list):
            if data_name is not None and load_fnc is not None:
                dataflow_list.append(data_name)
                self._load_fnc_list.append(load_fnc)
            else:
                break
        self._n_dataflow = len(dataflow_list)
        self._shuffle = shuffle
        self._batch_dict_name = batch_dict_name

        self._data_id = 0
        self.setup(epoch_val=0, batch_size=1)
        self._load_file_list(dataflow_list, data_dir)
        self._cur_file_name = [[] for i in range(len(self._file_name_list))]

    def setup(self, epoch_val, batch_size, **kwargs):
        # self.reset_epochs_completed(epoch_val)
        # self.set_batch_size(batch_size)
        # self.reset_state()
        self._epochs_completed  = epoch_val
        self._batch_size = batch_size
        self.rng = get_rng(self)
        self._setup()

    def _setup(self):
        pass

    def size(self):
        return len(self._file_name_list[0])

    def _load_file_list(self, data_name_list, data_dir_list):
        self._file_name_list = []
        for data_name, data_dir in zip(data_name_list, data_dir_list):
            self._file_name_list.append(get_file_list(data_dir, data_name))
        if self._shuffle:
            self._suffle_file_list()

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        for idx, file_list in enumerate(self._file_name_list):
            self._file_name_list[idx] = file_list[idxs]

    def next_batch(self):
       
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        if self._data_id + self._batch_size > self.size():
            start = self._data_id
            end = self.size()
        else:
            start = self._data_id
            self._data_id += self._batch_size
            end = self._data_id
        batch_data = self._load_data(start, end)

        for flow_id in range(len(self._file_name_list)):
            self._cur_file_name[flow_id] = self._file_name_list[flow_id][start: end]

        if end == self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        return batch_data

    def _load_data(self, start, end):
        data_list = [[] for i in range(0, self._n_dataflow)]
        for k in range(start, end):
            for read_idx, read_fnc in enumerate(self._load_fnc_list):
                data = read_fnc(self._file_name_list[read_idx][k])
                data_list[read_idx].append(data)

        for idx, data in enumerate(data_list):
            data_list[idx] = np.array(data)

        return data_list

    def next_batch_dict(self):
        batch_data = self.next_batch()
        return {key: data for key, data in zip(self._batch_dict_name, batch_data)} 

    def get_batch_file_name(self):
        return self._cur_file_name

    @property
    def epochs_completed(self):
        return self._epochs_completed
