# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import cv2
import numpy as np
from _utils.utils import *
from configure import config as cfg

class Generator:
    def __init__(self,
                 root_path: str,
                 batch_size: str,
                 train_steps: int,
                 validate_steps: int):

        self.warp = DataWarp()
        self.root_path = root_path
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.validate_steps = validate_steps
        self.pair_file_paths = Generator.get_file_paths(root_path)

    @classmethod
    def get_file_paths(cls, root_path):

        pair_file_paths = []
        for i, (root, dirs, files) in enumerate(os.walk(root_path)):
            if i:
                file_paths = []
                for file in files:
                    file_paths.append(os.path.join(root, file))
                pair_file_paths.append(file_paths)

        return pair_file_paths

    def generate(self, training: bool=True):

        while True:

            former_sources, latter_sources = [], []
            former_targets, latter_targets = [], []
            if training:
                steps = self.train_steps
            else:
                steps = self.validate_steps

            for step in range(steps):
                random_pair_paths = [np.random.choice(file_paths, size=self.batch_size, replace=False)
                                     for file_paths in self.pair_file_paths]

                for former_path, latter_path in zip(*random_pair_paths):

                    former_source = image_preprocess(former_path)
                    latter_source = image_preprocess(latter_path)

                    former_source = random_transform(former_source, **cfg.transform_kwargs)
                    former_source, former_target = self.warp(former_source)

                    latter_source = random_transform(latter_source, **cfg.transform_kwargs)
                    latter_source, latter_target = self.warp(latter_source)

                    former_sources.append(former_source)
                    former_targets.append(former_target)
                    latter_sources.append(latter_source)
                    latter_targets.append(latter_target)

                annotation_former_sources = np.array(former_sources.copy())
                annotation_former_targets = np.array(former_targets.copy())
                annotation_latter_sources = np.array(latter_sources.copy())
                annotation_latter_targets = np.array(latter_targets.copy())

                former_sources.clear()
                former_targets.clear()
                latter_sources.clear()
                latter_targets.clear()

                yield annotation_former_sources, annotation_former_targets, \
                      annotation_latter_sources, annotation_latter_targets

if __name__ == '__main__':

    generator = Generator(root_path=cfg.root_path,
                          batch_size=cfg.batch_size,
                          train_steps=cfg.train_steps,
                          validate_steps=cfg.validate_steps)

    train_gen = generator.generate(training=True)
    validate_gen = generator.generate(training=False)

    for i in range(generator.train_steps):

        former_sources, former_targets, latter_sources, latter_targets = next(train_gen)
        for former_source, latter_source in zip(former_targets, latter_targets):
            cv2.imshow('win1', cv2.cvtColor(former_source.astype("float32"), cv2.COLOR_RGB2BGR))
            cv2.imshow('win2', cv2.cvtColor(latter_source.astype("float32"), cv2.COLOR_RGB2BGR))
            cv2.waitKey(200)