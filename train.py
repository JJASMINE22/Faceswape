# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import tensorflow as tf
from faceswap import FaceSwape
from _utils.generate import Generator
from configure import config as cfg

if __name__ == '__main__':

    faceSwap = FaceSwape(betas=cfg.betas,
                         learning_rate=cfg.learning_rate)

    data_gen = Generator(root_path=cfg.root_path,
                          batch_size=cfg.batch_size,
                          train_steps=cfg.train_steps,
                          validate_steps=cfg.validate_steps)

    train_gen = data_gen.generate(training=True)
    validate_gen = data_gen.generate(training=False)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(model=faceSwap.model,
                               optimizer=faceSwap.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    for epoch in range(cfg.Epoches):
        # ----training----
        print('------start training------')
        for i in range(data_gen.train_steps):
            former_sources, former_targets, latter_sources, latter_targets = next(train_gen)
            faceSwap.train(former_sources, former_targets,
                           latter_sources, latter_targets)
            if not (i + 1) % cfg.per_sample_interval:
                faceSwap.generate_sample(former_sources, latter_sources, i + 1)

        print(f'Epoch: {epoch + 1}\n',
              f'train_loss: {faceSwap.train_loss.result().numpy()}\n')

        # ----validating----
        print('------start validating------')
        for i in range(data_gen.validate_steps):
            former_sources, former_targets, latter_sources, latter_targets = next(validate_gen)
            faceSwap.validate(former_sources, former_targets,
                              latter_sources, latter_targets)

        print(f'Epoch: {epoch + 1}\n',
              f'validate_loss: {faceSwap.validate_loss.result().numpy()}\n')

        ckpt_save_path = ckpt_manager.save()

        faceSwap.train_loss.reset_states()
        faceSwap.validate_loss.reset_states()
