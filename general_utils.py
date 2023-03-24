import os
import numpy as np
import pandas as pd
import glob as gb
import tensorflow as tf
from tensorflow import keras
from seq2mat import seq2mat, seq2mat_batch, seq2img, seq2img_aio
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pre_mobnet


all_classes = ['IGHV3-23', 'IGHV1-69-2', 'IGHV6-1', 'IGHV3-30',
               'IGHV1-46', 'IGHV3-11', 'IGHV1-24', 'IGHV5-10-1',
               'IGHV1-2', 'IGHV7-4-1','IGHV1-18', 'IGHV1-69D',
               'IGHV1-8', 'IGHV1-69', 'IGHV3-33', 'IGHV5-51',
               'IGHV4-34', 'IGHV3-53', 'IGHV2-70', 'IGHV2-5',
               'IGHV3-21', 'IGHV2-26', 'IGHV1-3', 'IGHV3-9'
               ]
all_classes_perc = [0.031, 0.016, 0.094, 0.059,
                    0.071, 0.014, 0.067, 0.029,
                    0.052, 0.011, 0.076, 0.031,
                    0.051, 0.112, 0.026, 0.059,
                    0.048, 0.011, 0.015, 0.021,
                    0.030, 0.021, 0.012, 0.013
                    ]

general_classes = ['IGHV1', 'IGHV2', 'IGHV3', 'IGHV4', 'IGHV5', 'IGHV6']  # , 'IGHV7']

temp1, temp2, temp3 = [], [], []
for i in range(len(all_classes)):
    idx = np.argmax(all_classes_perc)
    temp1.append(all_classes[idx])
    temp2.append(all_classes_perc[idx])
    temp3.append((all_classes[idx], all_classes_perc[idx]))
    all_classes.pop(idx)
    all_classes_perc.pop(idx)

all_classes, all_classes_prc = temp1.copy(), temp2.copy()


def seqlbl_extraction(paths, inc_subsets, exc_subsets, shuffle=True, seed=10):
    """
    ،This function is to extract sequences and their labels from txt files
    :param paths: the list of path of txt files
    :param subsets: the list of subsets which have to be classified
    :param shuffle:
    :param seed:
    :return: sequences and their labels
    """
    seq_list, lbl_list = [], []
    for path in paths:
        file = pd.read_table(path)
        cdrh3 = file.aaSeqCDR3
        vgenes = file.bestVGene
        for i, seq in enumerate(cdrh3):
            if '*' in seq or '_' in seq:
                continue
            elif len(seq) < 8 or len(seq) > 28:
                # elif len(seq) != 15:
                continue
            elif vgenes[i] not in all_classes:
                continue
            elif vgenes[i] in exc_subsets:
                continue
            else:
                seq_list.append(seq)
                temp = vgenes[i]
                try:
                    lbl_list.append(inc_subsets.index(temp))
                except:
                    lbl_list.append(len(inc_subsets))

    seq_list_new, lbl_list_new = [], []
    if shuffle:
        np.random.seed(seed)
        per_list = np.random.permutation(len(seq_list))
        for idx in per_list:
            seq_list_new.append(seq_list[idx])
            lbl_list_new.append(lbl_list[idx])
        return seq_list_new, lbl_list_new

    return seq_list, lbl_list


def seqlbl_extraction_general(paths, shuffle=True, seed=10):
    """
    ،This function is to extract sequences and their labels from txt files
    :param paths: the list of path of txt files
    :param subsets: the list of subsets which have to be classified
    :param shuffle:
    :param seed:
    :return: sequences and their labels
    """
    seq_list, lbl_list = [], []
    for path in paths:
        file = pd.read_table(path)
        cdrh3 = file.aaSeqCDR3
        vgenes = file.bestVGene
        for i, seq in enumerate(cdrh3):
            if '*' in seq or '_' in seq:
                continue
            elif len(seq) < 8 or len(seq) > 28:
                # elif len(seq) != 15:
                continue
            elif vgenes[i] not in all_classes:
                continue
            else:
                try:
                    temp = vgenes[i]
                    idx = temp.find('-')
                    general_type = temp[:idx]
                    lbl = general_classes.index(general_type)
                    lbl_list.append(lbl)
                    seq_list.append(seq)
                except:
                    pass

    seq_list_new, lbl_list_new = [], []
    if shuffle:
        np.random.seed(seed)
        per_list = np.random.permutation(len(seq_list))
        for idx in per_list:
            seq_list_new.append(seq_list[idx])
            lbl_list_new.append(lbl_list[idx])
        return seq_list_new, lbl_list_new

    return seq_list, lbl_list


def under_sampling(x, y, threshold = 5000, seed=10):
    """
    Under sampling method to struggle with imbalanced data
    :param x: input samples
    :param y: labels
    :param threshold: maximum number of samples in each class
    :param seed:
    :return: inputs and outputs with reduction samples
    """
    num_classes = np.max(y) + 1
    x_sep = []
    counter = [0]*num_classes
    for i in range(num_classes):
        temp = []
        for j, lbl in enumerate(y):
            if lbl==i:
                temp.append(x[j])

        x_sep.append(temp)

    x_u_sampling, y_u_sampling = [], []
    for i, samples in enumerate(x_sep):
        print(i, len(samples))
        if len(samples) > threshold:
            np.random.seed(seed)
            samples = np.random.permutation(samples)
            samples = samples[:threshold]
        x_u_sampling.extend(samples)
        y_u_sampling.extend([i]*len(samples))
    np.random.seed(seed)
    x_u_sampling = np.random.permutation(x_u_sampling)
    np.random.seed(seed)
    y_u_sampling = np.random.permutation(y_u_sampling)
    return x_u_sampling, y_u_sampling


def over_sampling(x, y, subsets, multiplication, seed=42):
    x_new, y_new = [], []
    for i, lbl in enumerate(y):
        seq = x[i]
        if lbl in subsets:
            idx = subsets.index(lbl)
            mltpl = multiplication[idx]
            x_temp = [seq]*mltpl
            y_temp = [lbl]*mltpl
            x_new.extend(x_temp)
            y_new.extend(y_temp)
        else:
            x_new.append(seq)
            y_new.append(lbl)
    x_new_sh, y_new_sh = [], []
    np.random.seed(seed)
    list_per = np.random.permutation(len(x_new))
    for idx in list_per:
        x_new_sh.append(x_new[idx])
        y_new_sh.append(y_new[idx])
    return x_new, y_new


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, outputs, func, batch_size=8, shuffle=True):
        'Initialization'

        self.batch_size = batch_size
        self.labels = outputs
        self.inputs = inputs
        self.func = func
        self.n_classes = int(np.max(outputs))+1
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        inputs_temp = [self.inputs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(inputs_temp, labels_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, inputs, labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = self.func(inputs)
        y = labels
        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)


