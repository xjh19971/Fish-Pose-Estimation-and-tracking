﻿import os
import re
import sys

import keras.backend as K
import pandas
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

K.set_session(sess)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from keras.layers.convolutional import Conv2D
from model.cmu_model_DenseNet_NotPre_RESIZE_Loss import get_training_model
from training.dataset import get_dataflow, batch_dataflow

batch_size = 16
base_lr = 1e-2  # 2e-5
weight_decay = 5e-4
max_iter = 3000  # 600000

weights_best_file = "weights.best_final_train.h5"
training_log = "training.csv"
logs_dir = "./logs"

from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def get_last_epoch():
    """
    Retrieves last epoch from log file updated during training.

    :return: epoch number
    """
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)


def restore_weights(weights_best_file, model):
    """
    Restores weights from the checkpoint file if exists or
    preloads the first layers with VGG19 weights

    :param weights_best_file:
    :return: epoch number to use to continue training. last epoch + 1 or 0
    """
    # load previous weights or vgg19 if this is the first run
    if os.path.exists(weights_best_file):
        print("Loading the best weights...")

        model.load_weights(weights_best_file)

        return get_last_epoch() + 1
    else:
        '''print("Loading VGG weights...")

        vgg_model = VGG19(include_top=False,weights='imagenet')

        for layer in model.layers:
            if layer.name in from_vgg:
                vgg_layer_name = from_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())

        print("Loaded VGG layer")
        '''
        print("Loading mobilenet weights...")
        # base_model=MobileNetV2(include_top=False,weights='imagenet')
        # WEIGHTS_PATH='./mobilenetv2_weight.h5'
        # base_model.save_weights(WEIGHTS_PATH)
        # model.load_weights(WEIGHTS_PATH,by_name=True)
        print("Loaded mobilenet layer")
        return 0


def get_lr_multipliers(model):
    """
    Setup multipliers for stageN layers (kernel and bias)

    :param model:
    :return: dictionary key: layer name , value: multiplier
    """
    lr_mult = dict()
    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                lr_mult[kernel_name] = 1

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                lr_mult[kernel_name] = 4

            # vgg
            else:
                kernel_name = layer.weights[0].name
                lr_mult[kernel_name] = 1

    return lr_mult


def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """

    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["Output_1_1"] = _eucl_loss
    losses["Output_1_2"] = _eucl_loss
    losses["Output_2_1"] = _eucl_loss
    losses["Output_2_2"] = _eucl_loss
    losses["Output_3_1"] = _eucl_loss
    losses["Output_3_2"] = _eucl_loss
    losses["Output_4_1"] = _eucl_loss
    losses["Output_4_2"] = _eucl_loss
    losses["Output_5_1"] = _eucl_loss
    losses["Output_5_2"] = _eucl_loss

    return losses


def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i


if __name__ == '__main__':

    # get the model

    model = get_training_model(weight_decay)
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    # restore weights

    last_epoch = restore_weights(weights_best_file, model)

    # prepare generators

    curr_dir = os.path.dirname(__file__)
    train_annot_path = os.path.join(curr_dir, '../dataset/my_keypoints_final_train.json')
    train_img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/train_tracking_data/'))
    val_annot_path = os.path.join(curr_dir, '../dataset/my_keypoints_final_test.json')
    val_img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/train_tracking_data/'))
    # get dataflow of samples

    df1 = get_dataflow(
        annot_path=train_annot_path,
        img_dir=train_img_dir)
    train_samples = df1.size()
    df2 = get_dataflow(
        annot_path=val_annot_path,
        img_dir=train_img_dir)
    val_samples = df2.size()
    # get generator of batches

    batch_df1 = batch_dataflow(df1, batch_size)
    train_gen = gen(batch_df1)
    batch_df2 = batch_dataflow(df2, batch_size)
    val_gen = gen(batch_df2)
    # setup lr multipliers for conv layers

    lr_multipliers = get_lr_multipliers(model)

    # configure callbacks

    iterations_per_epoch = train_samples // batch_size
    lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=30, mode='auto')
    checkpoint = ModelCheckpoint(weights_best_file, monitor='val_loss',
                                 verbose=0, save_best_only=True,
                                 save_weights_only=True, mode='min', period=1)
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                     write_images=False)

    callbacks_list = [lrate, checkpoint, csv_logger, tb]

    # sgd optimizer with lr multipliers

    # multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
    #                   nesterov=False, lr_mult=lr_multipliers)

    # start training
    adam = optimizers.Adam(lr=base_lr)
    loss_funcs = get_loss_funcs()
    model.compile(loss=loss_funcs, optimizer=adam)
    model.fit_generator(train_gen,
                        steps_per_epoch=train_samples // batch_size,
                        epochs=max_iter,
                        callbacks=callbacks_list,
                        validation_data=val_gen,
                        validation_steps=val_samples // batch_size,
                        use_multiprocessing=False,
                        initial_epoch=last_epoch)
