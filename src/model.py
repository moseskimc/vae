import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, layers


class Sampler(layers.Layer):
    
    def call(self, inputs):
        """Return reparam. of latent z and thresh of stand dev of post. dist.

        Args:
            inputs (tuple): mean and sd of latent vector z from encoder

        Returns:
            tuple: tuple of np.ndarray objects 
                   reparameterized z = mean + sd.epsilon, epsilon ~ N(0,I) and
                   thresholded standard deviation of posterior distribution
        """
        # posterior distribution from encoder (deterministic)
        z_mean, z_stdev = inputs
        
        # extract batch size and dimension of latent space
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # ensure non-negative sd
        sd_gt_0 = tf.math.log(1 + tf.math.exp(z_stdev))  
        
        # reparameterize
        # epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        epsilon = tf.random.normal(shape=(batch_size, dim))
        z_sample = z_mean + sd_gt_0 * epsilon
        return z_sample, sd_gt_0
        

class Encoder(layers.Layer):
    
    def __init__(self, input_shape, latent_dim, name="encoder", **kwargs):
        """Encoder layer

        Args:
            input_shape (tuple): dimensions of single sample
            latent_dim (int): latent space dimension
            name (str, optional): name of layer; defaults to "encoder".
        """
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        
        # input dims
        self.encoder_input_shape = input_shape
        
        # layers
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, activation="relu", strides=(2,2), padding="same")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, activation="relu", strides=(2,2), padding="same")
        self.flatten = layers.Flatten()
        # gauss dist
        self.dense_mean = layers.Dense(self.latent_dim, activation=None)
        self.dense_stdev = layers.Dense(self.latent_dim, activation=None)        
        self.sampler = Sampler()

    def call(self, x):
        """Feed forward
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        # create distribution
        z_mean = self.dense_mean(x)
        z_stdev = self.dense_stdev(x)
        
        # extract reparameterization (z)
        # as well as stddev of posterior (sd)
        z, sd = Sampler()((z_mean, z_stdev))
        
        return z, z_mean, sd

class Decoder(layers.Layer):
    
    def __init__(self, latent_dim, name="decoder", **kwargs):
        """Decoder layer

        Args:
            latent_dim (int): dimension of latent space
            name (str, optional): name of layer; defaults to "decoder".
        """
        super(Decoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        # layers
        self.dense = layers.Dense(7*7*32, activation=None)
        self.reshape = layers.Reshape((7,7,32))
        self.deconv1 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", activation="relu")
        self.deconv2 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")
        self.deconv3 = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same")
    
    def call(self, x):
        """Feed forward
        """
        x = self.dense(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class VAE(Model):
    
    def __init__(self, input_dims, latent_dim, kl_weight=1, name="autoencoder", **kwargs):
        """Variational Autoencoder Model

        Args:
            input_dims (tuple): dimensions of single image sample
            latent_dim (int): latent space dimension
            kl_weight (int, optional): Kullback-Leibler weight; defaults to 1.
            name (str, optional): model name; defaults to "autoencoder".
        """
        super(VAE, self).__init__(name=name, **kwargs)
        # model params
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        # encoder and decoder
        self.encoder = Encoder(input_dims, latent_dim)
        self.decoder = Decoder(latent_dim)

    def call(self, x):
        """Feed forward and compute KL loss
        """
        # posterior distribution
        # and reparameterization
        z_sample, z_mean, z_sd = self.encoder(x)
        x_recons_logits = self.decoder(z_sample)
        
        # compute kl loss
        kl_loss = (-1.0/self.latent_dim)*tf.math.reduce_sum(1 + tf.math.log(tf.math.square(z_sd)) - 
                                          tf.math.square(z_mean) - 
                                          tf.math.square(z_sd), axis=1)
        kl_loss = tf.math.reduce_mean(kl_loss)
        kl_loss = self.kl_weight * kl_loss
        self.add_loss(kl_loss)

        return x_recons_logits
    
    
def reconstruction_loss(x_input, model):
    """Return mean cross entropy loss from logits

    Args:
        x_input (np.ndarray): batch of images
        model (tf.keras.Model): autoencoder model

    Returns:
        float: batch mean of reduced sums of pixel-wise sigmoid 
               cross entropy loss
    """
    
    recons = model(x_input)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_input, logits=recons)
    neg_log_lklhd = tf.math.reduce_sum(cross_entropy, axis=[1,2,3])
    return tf.math.reduce_mean(neg_log_lklhd)


# batch train step
@tf.function
def vae_train_step(x_input, model, optimizer, loss_metric):
    """Train on a given batch updating gradients using an optimizer

    Args:
        x_input (np.ndarray): batch of sample images
        model (tf.keras.Model): autoencoder model object
        optimizer (tf.keras.optimizers.Optimizer): adam or sgd optimizer object
        loss_metric (tf.keras.metrics.Metric): loss metric object
    """

    with tf.GradientTape() as tape:
        recon_loss = reconstruction_loss(x_input, model)
        kl_loss = tf.math.reduce_sum(model.losses)
        # add both losses
        loss = recon_loss + kl_loss
        
    # update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # update loss function
    loss_metric(loss)
    
# eval step
@tf.function
def vae_eval_step(x_input, model, loss_metric):
    """Update validation loss metric

    Args:
        x_input (np.ndarray): batch of sample images
        model (tf.keras.Model): autoencoder model object
        loss_metric (tf.keras.metrics.Metric): metric object
    """
    recon_loss = reconstruction_loss(x_input, model)  # NOTE: need to be caerful if weights are trained during evaluation
    kl_loss = tf.math.reduce_sum(model.losses)
    
    loss = recon_loss + kl_loss
    loss_metric(loss)

    
# preprocess images for ease of enconding
def vae_preprocess_images(images):
    """Return after reshaping, normalizing, and thresholding

    Args:
        images (np.ndarray): batch of sample images

    Returns:
        np.ndarray: pre-processed batch of images
    """
    # reshape
    images = images.reshape((images.shape[0], 28, 28, 1))
    # normalize
    images = images / 255
    # threshold (for reconstruction step)
    images = np.where(images > 0.5, 1.0, 0.0).astype('float32')
    
    return images
