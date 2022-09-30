import numpy as np
import tensorflow as tf
from PIL import Image
from net.big_scale_network import CreateModel
from configure import config as cfg

class FaceSwape:
    def __init__(self,
                 betas: dict,
                 learning_rate: float,
                 **kwargs):

        self.model = CreateModel()

        self.loss_func = tf.keras.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam(**betas,
                                                  learning_rate=learning_rate)

        self.train_loss = tf.metrics.Mean()
        self.validate_loss = tf.metrics.Mean()

    @tf.function
    def train(self, former_sources, former_targets, latter_sources, latter_targets):

        with tf.GradientTape(persistent=True) as tape:
            former_logits = self.model(former_sources, mode='former')
            latter_logits = self.model(latter_sources, mode='latter')
            former_loss = self.loss_func(former_targets, former_logits)
            latter_loss = self.loss_func(latter_targets, latter_logits)
            loss = former_loss + latter_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def validate(self, former_sources, former_targets, latter_sources, latter_targets):

        former_logits = self.model(former_sources, mode='former')
        latter_logits = self.model(latter_sources, mode='latter')
        former_loss = self.loss_func(former_targets, former_logits)
        latter_loss = self.loss_func(latter_targets, latter_logits)
        loss = former_loss + latter_loss

        self.validate_loss(loss)

    def generate_sample(self, former_sources, latter_sources, batch):

        random_index = np.random.choice(former_sources.shape[0], size=(1,))
        former_source = former_sources[random_index]
        latter_source = latter_sources[random_index]

        former_logit = self.model(former_source, mode='former').numpy()
        latter_logit = self.model(latter_source, mode='latter').numpy()
        former_fake = self.model(latter_source, mode='former').numpy()
        latter_fake = self.model(former_source, mode='latter').numpy()

        former_sample = np.concatenate([former_source.squeeze(axis=0),
                                        former_logit.squeeze(axis=0),
                                        latter_fake.squeeze(axis=0)],
                                       axis=1)

        latter_sample = np.concatenate([latter_source.squeeze(axis=0),
                                        latter_logit.squeeze(axis=0),
                                        former_fake.squeeze(axis=0)],
                                       axis=1)

        former_sample = np.clip(former_sample, -1., 1.)
        latter_sample = np.clip(latter_sample, -1., 1.)

        former_sample = (former_sample + 1) * 127.5
        latter_sample = (latter_sample + 1) * 127.5

        former_sample = Image.fromarray(former_sample.astype('uint8'))
        latter_sample = Image.fromarray(latter_sample.astype('uint8'))

        former_sample.save(cfg.former_sample_path.format(batch), quality=95, subsampling=0)
        latter_sample.save(cfg.latter_sample_path.format(batch), quality=95, subsampling=0)
