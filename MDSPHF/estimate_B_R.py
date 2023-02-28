# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:58:30 2021

@author: Dunbin Shen
"""

import tensorflow as tf
import os
import numpy as np
from skimage.measure import compare_psnr
import random
import scipy.io as sio
from utils.quality_measure import SAM
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def spectralDegrade(X, R, addNoise=True, SNR=40):
    '''
    spectral downsample
    :param X:
    :param R:
    :return:
    '''
    height, width, bands = X.shape
    X = np.reshape(X, [-1, bands], order='F')
    Z = np.dot(X, R.T)
    Z = np.reshape(Z, [height, width, -1], order='F')

    if addNoise:
        h, w, c = Z.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Z)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Z += sigmah * np.random.randn(h, w, c)

    return Z


def downSample(X, B, ratio, addNoise=True, SNR=30):
    '''
    downsample using fft
    :param X:
    :param B:
    :param ratio:
    :return:
    '''
    B = np.expand_dims(B, axis=-1)
    Y = np.real(np.fft.ifftn(np.fft.fftn(X) * B))

    if addNoise:
        h, w, c = Y.shape
        numl = h * w * c
        sigmah = np.sqrt(np.sum(np.square(Y)) / pow(10, SNR / 10) / numl)
        print(sigmah)
        Y += sigmah * np.random.randn(h, w, c)

    # downsample
    Y = Y[::ratio, ::ratio, :]
    return Y


def fusion_net(Y, Z, num_spectral=31, ms_num_spectral=3, ratio=4, h=64, w=64,
               reuse=False):
    with tf.variable_scope('fusion_net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        kernel_size = 7

        # Define the blurring kernel
        with tf.variable_scope('p1'):
            filter = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size], stddev=0.1), name='filter')

        # The sum-to-one constraint
        # filter = tf.exp(filter)
        filter = filter / tf.reduce_sum(filter)

        # Get the cicular matrix in which the parameters are located in the four corners
        part_1 = tf.zeros([h - kernel_size, kernel_size], dtype=tf.float32)
        part_2 = tf.zeros([h, w - kernel_size], dtype=tf.float32)
        B_tensor = tf.concat([filter, part_1], axis=0)
        B_tensor = tf.concat([B_tensor, part_2], axis=1)
        B_tensor = tf.roll(tf.roll(B_tensor, -kernel_size // 2, axis=0), -kernel_size // 2, axis=1)

        # FFT
        B_tensor = tf.cast(B_tensor, tf.complex64)
        B_tensor = tf.fft2d(B_tensor)

        B_saved = B_tensor

        # Expand the dimension to facilitate the multiplication int fourier space
        B_tensord = tf.expand_dims(B_tensor, axis=-1)
        B_tensor = tf.tile(B_tensord, [1, 1, ms_num_spectral])

        # Define the spectral degradation matrix
        with tf.variable_scope('p1'):
            R = tf.Variable(tf.truncated_normal(shape=[num_spectral, ms_num_spectral], stddev=0.1), name='rsf')

        # The sum-to-one constrainst in each row
        div = tf.reduce_sum(R, axis=0)
        # div = tf.nn.softmax(R,axis=0)
        div = tf.expand_dims(div, axis=0)
        R = R / div
        # R = tf.nn.softmax(R,axis=0)
        R_tensor = R

        # Expand the dimension to facilitate the matrix multiplication
        R = tf.expand_dims(R, axis=0)
        R = tf.tile(R, [tf.shape(Z)[0], 1, 1])

        # The blurring operation by element multiplication in fourier space and convert to the real space
        Z_tensor = tf.cast(Z, tf.complex64)
        ZC = tf.real(tf.ifft3d(tf.fft3d(Z_tensor) * B_tensor))

        # The downsampling operation
        ZC = ZC[:, ::ratio, ::ratio, :]

        # The spectral degradation
        Y_shaped = tf.reshape(Y, [-1, (h // ratio) * (w // ratio), num_spectral])
        RY = tf.matmul(Y_shaped, R)
        RY = tf.reshape(RY, [-1, h // ratio, w // ratio, ms_num_spectral])

        return filter, B_saved, R_tensor, RY, ZC


def produce_random_list(random_list: list, total, k):
    '''
    产生k个1-total间不重复的随机整数
    :param random_list:
    :param total:
    :param k:
    :return:
    '''
    i = 0
    while i < k:
        n = random.randint(1, total)
        if n not in random_list:
            random_list.append(n)
            i += 1


def load_train_batch(train_piece_num, train_batch_size, size_x, size_y, bands, msbands, ratio, path):
    '''
    随机加载批量训练数据
    :return:
    '''
    helplist = []
    train_lrhs = np.zeros([train_batch_size, size_x // ratio, size_y // ratio, bands], dtype=np.float32)
    train_ms = np.zeros([train_batch_size, size_x, size_y, msbands], dtype=np.float32)
    produce_random_list(helplist, train_piece_num, train_batch_size)
    for inx, value in enumerate(helplist):
        mat = sio.loadmat(path + '%d.mat' % value)
        train_lrhs[inx, :, :, :] = mat['Y']
        train_ms[inx, :, :, :] = mat['Z']
    helplist.clear()
    return train_lrhs, train_ms


def train(num=0):
    '''

    :param num: the NO. of the dataset
    0 ----- CAVE
    1 ----- Harvard
    2 ----- Pavia Unversity
    3 ----- University of Houstan
    4 ----- World View-2
    :return:
    '''
    start = time.perf_counter()
    psnr_max = 15
    lr = 1e-2

    if num == 0:
        model_directory = r'models/B_R/CAVE_train/'
        bands = 31
        ms_bands = 3
        ratio = 8
        h = 512
        w = 512

        batch_size = 1
        total_num = 32

        num_start = 21
        num_end = 32

        data_path = r'CAVEMAT/'
    ############## placeholder for training
    ms = tf.placeholder(dtype=tf.float32, shape=[None, None, None, ms_bands])
    lrhs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, bands])
    ######## network architecture

    _, _, _, RY, ZC = fusion_net(lrhs, ms, num_spectral=bands, ms_num_spectral=ms_bands, h=h, w=w, ratio=ratio)

    IM = tf.image.resize_bicubic(ms, (h // ratio, w // ratio), name='bicubic')

    loss = tf.reduce_mean(tf.abs(RY - ZC)) + 0.1 * tf.reduce_mean(tf.abs(RY - IM))

    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fusion_net/p1')

    f_optim = tf.train.AdamOptimizer(lr, beta1=0.9) \
        .minimize(loss, var_list=t_vars)

    ##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)  # 2979432

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        i = 0
        icount = 0

        while i < 1000:
            train_y, train_z = load_train_batch(total_num, batch_size, h, w, bands, ms_bands, ratio,
                                                data_path)

            _, mse_loss = sess.run([f_optim, loss],
                                   feed_dict={lrhs: train_y, ms: train_z})
            icount += 1

            if icount * batch_size == total_num:
                icount = 0
                i += 1
                V1, V2 = sess.run([RY, ZC], feed_dict={lrhs: train_y, ms: train_z})
                # ZC_val = np.clip(ZC_val,0,1)
                V1 = np.clip(V1, 0, 1)
                V2 = np.clip(V2, 0, 1)
                psnr = compare_psnr(V1, V2)

                # sam = sam * 180 / np.pi
                print("Epoch: " + str(i) + " MSE: " + str(mse_loss) + " PSNR: " + str(psnr))
                psnr = 0
                for j in range(num_start, num_end + 1):
                    mat = sio.loadmat(data_path + '%d.mat' % j)
                    valid_y = mat['Y']
                    valid_z = mat['Z']
                    # valid_p = mat['P']
                    # valid_x = mat['label']
                    valid_y = np.expand_dims(valid_y, 0)
                    valid_z = np.expand_dims(valid_z, 0)
                    # valid_x = np.expand_dims(valid_x,0)
                    V1, V2 = sess.run([RY, ZC], feed_dict={lrhs: valid_y, ms: valid_z})
                    # ZC_val2 = np.clip(ZC_val2,0,1)
                    V1 = np.clip(V1, 0, 1)
                    V2 = np.clip(V2, 0, 1)
                    psnr += compare_psnr(V1, V2)
                tt = psnr / (num_end - num_start + 1)

                if tt > psnr_max:
                    psnr_max = tt
                    saver.save(sess, model_directory + '-' + str(i) + '.ckpt')
                    print('valid PSNR:%s' % tt)
                    print("Save Model")

        end = time.perf_counter()
        print('用时%ss' % (end - start))

        coord.request_stop()
        coord.join(threads)
        sess.close()


def test(num=0):
    '''

    :param num: the NO. of the dataset
    0 ----- CAVE
    1 ----- Harvard
    2 ----- Pavia Unversity
    3 ----- University of Houstan
    4 ----- World View-2
    5 ----- HypSen
    :return:
    '''
    start = time.perf_counter()
    if num == 0:
        model_directory = r'models/B_R/CAVE_train/'
        bands = 31
        ms_bands = 3
        ratio = 8
        h = 512
        w = 512
        num_end = 32
        num_start = 21
        data_path = r'CAVEMAT/'

    ms = tf.placeholder(dtype=tf.float32, shape=[None, None, None, bands])
    pan = tf.placeholder(dtype=tf.float32, shape=[None, None, None, ms_bands])

    B_filter, BT, RT, RY, ZC = fusion_net(ms, pan, h=h, w=w, num_spectral=bands, ms_num_spectral=ms_bands, ratio=ratio)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if tf.train.get_checkpoint_state(model_directory):
            latest_model = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess, latest_model.model_checkpoint_path)
            print("load model")

        psnr = 0
        sam = 0
        B, B1, R1 = sess.run([B_filter, BT, RT])
        R1 = R1.T
        # sio.savemat(model_directory + 'R_B.mat', {'R': R1, 'B': B})
        # mat = sio.loadmat('saved_B_R/CAVE/R_B.mat')
        # B1 = mat['B']
        # R1 = mat['R']
        for i in range(num_start, num_end + 1):
            print(i)
            mat = sio.loadmat(data_path + '%d.mat' % i)
            test_Z = mat['Z']
            test_Y = mat['Y']

            test_Z = np.clip(test_Z, 0, 1)

            V1 = spectralDegrade(test_Y, R1, addNoise=False)
            V2 = downSample(test_Z, B1, ratio, False)
            V3 = test_Z[::ratio, ::ratio, :]

            V1 = np.clip(V1, 0, 1)
            V2 = np.clip(V2, 0, 1)
            V3 = np.clip(V3, 0, 1)

            psnr += compare_psnr(V1, V2)
            sam += (SAM(V3, V2)[0])

        print('Our PSNR and SAM')
        print(psnr / (num_end - num_start + 1))
        print(sam / (num_end - num_start + 1))

        end = time.perf_counter()
        print('用时%ss' % ((end - start) / (num_end - num_start + 1)))


if __name__ == '__main__':
    train(0)
    tf.reset_default_graph()
    test(0)
