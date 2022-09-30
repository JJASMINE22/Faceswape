# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import cv2
import numpy as np
import tensorflow as tf
from configure import config as cfg

def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num

    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def image_preprocess(file_path, cvt=cv2.cvtColor):

    image = cv2.resize(cv2.imread(file_path), cfg.image_size)
    if cvt: image = cvt(image, cv2.COLOR_BGR2RGB)
    image = image/127.5 - 1.
    image = np.clip(image, -1., 1.)

    return image

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


class DataWarp:
    def __init__(self,
                 scale_size: tuple=(5, 5),
                 input_shape: tuple = (128, 128),
                 padding_size: tuple = (16, 16)):
        self.scale_size = scale_size
        self.input_shape = input_shape
        self.padding_size = padding_size

    def __call__(self, image: np.ndarray, mode: str = 'tf'):
        assert image.shape.__len__() == 3

        # optional, crop key areas of face
        image = image[48: 208, 48:208]

        image = cv2.resize(image, self.input_shape)
        # 3 Dimension
        if mode == 'tf':
            image = tf.pad(image, tf.constant([self.padding_size,
                                               self.padding_size,
                                               [0, 0]])).numpy()
        else:
            image = np.concatenate(
                [np.zeros(shape=(self.input_shape[0], self.padding_size[0], 3), dtype='uint8'), image,
                 np.zeros(shape=(self.input_shape[0], self.padding_size[0], 3), dtype='uint8')], axis=1)

            image = np.concatenate([np.zeros(shape=(self.padding_size[1],
                                                    self.input_shape[1] + sum(self.padding_size), 3), dtype='uint8'),
                                    image,
                                    np.zeros(shape=(self.padding_size[1],
                                                    self.input_shape[1] + sum(self.padding_size), 3), dtype='uint8')],
                                   axis=0)

        gridx = np.linspace(self.padding_size[1], self.input_shape[1] + self.padding_size[1], self.scale_size[1])
        gridy = np.linspace(self.padding_size[0], self.input_shape[0] + self.padding_size[0], self.scale_size[0])

        gridy, gridx = np.meshgrid(gridy, gridx, indexing='ij')
        gridx += np.random.normal(size=(self.scale_size[1],) * 2, scale=5)
        gridy += np.random.normal(size=(self.scale_size[0],) * 2, scale=5)

        gridx = cv2.resize(gridx, (self.input_shape[1] + 4 * self.padding_size[1],) * 2)
        gridy = cv2.resize(gridy, (self.input_shape[0] + 4 * self.padding_size[0],) * 2)
        gridx = np.clip(gridx, self.padding_size[1], self.input_shape[1] + self.padding_size[1] - 1)
        gridy = np.clip(gridy, self.padding_size[0], self.input_shape[0] + self.padding_size[0] - 1)

        interp_gridx = gridx[2 * self.padding_size[1]: self.input_shape[1] + 2 * self.padding_size[1],
                             2 * self.padding_size[1]: self.input_shape[1] + 2 * self.padding_size[1]].astype('float32')
        interp_gridy = gridy[2 * self.padding_size[0]: self.input_shape[0] + 2 * self.padding_size[0],
                             2 * self.padding_size[0]: self.input_shape[0] + 2 * self.padding_size[0]].astype('float32')

        warped_image = cv2.remap(image, interp_gridx, interp_gridy, cv2.INTER_LINEAR)

        source_image = image[self.padding_size[0]: self.input_shape[0] + self.padding_size[0],
                             self.padding_size[1]: self.input_shape[1] + self.padding_size[1]]

        return warped_image, source_image
