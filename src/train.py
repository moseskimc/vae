import tensorflow as tf
import numpy as np

import time
import yaml

from tqdm import tqdm

from src.model import VAE, train_step, eval_step, preprocess_images


if __name__ == "__main__":

    tf.executing_eagerly()

    # load config file for training
    config = yaml.safe_load(open('config.yml'))

    # training params
    batch_size = config['batch_size']
    no_epochs = config['epochs']
    input_dims = eval(config['input_shape'])
    no_classes = config['num_classes']
    latent_dim = config['latent_dim']

    # initialize model
    model = VAE(input_dims=input_dims, latent_dim=latent_dim)
    
    # instantiate optimizer
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    # log to tensorboard
    if config['tensorboard']:
        writer1 = tf.summary.create_file_writer('logs/train')
        writer2 = tf.summary.create_file_writer('logs/eval')
        

    # Load mnist fashion data (test data)
    (_, _), (x_test_orig, y_test_orig) = tf.keras.datasets.fashion_mnist.load_data()

    # perform a 7:3 split
    train_len = x_test_orig.shape[0]
    test_len = int(0.3 * train_len)
    x_train, y_train = x_test_orig[:-test_len:, :, :], y_test_orig[:-test_len]
    x_test, y_test = x_test_orig[-test_len:, :, :], y_test_orig[-test_len:]
    # preprocess images
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    
    # instantiate data generators for train
    train_size = x_train.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(config['batch_size'])
    # for test
    test_size = x_test.shape[0]
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(config['batch_size'])

        
    # start training
    loss_history = []  
    best_val_acc = 0      # for classification
    best_loss = -np.inf   # autoencoder loss
    
    # define keras tensorboard callback
    tf.compat.v1.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
                                             write_grads=False, write_images=False, embeddings_freq=0,
                                             embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                             update_freq='epoch', profile_batch=2)
    
    for epoch in range(config['epochs']):
        # update learning rate
        if epoch in eval(config['learning_rate_steps']):
            optimizer.lr.assign(optimizer.lr.read_value() * config['learning_rate_decay'])

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        print('Training...')
        # train epoch
        for batch, (images, _) in enumerate(tqdm(dataset, total=train_size // config['batch_size'])):
            train_step(images, model, optimizer, epoch_loss_avg)
        time.sleep(0.5)
        print('Evaluating...')
        # evaluate epoch
        for batch, (images, _) in enumerate(tqdm(test_dataset, total=test_size // config['batch_size'])):
            eval_step(images, model, epoch_val_loss_avg)
        time.sleep(0.5)
        # compute elbo (evidence lower bound loss)
        elbo_loss = -1 * epoch_loss_avg.result()
        elbo_val_loss = -1 * epoch_val_loss_avg.result()
        
        # store values on tensorboard
        with writer1.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
        with writer2.as_default():
            tf.summary.scalar('loss', epoch_val_loss_avg.result(), step=epoch)
        
        # print epoch metrics
        print("Epoch {:03d} - loss: {:.3f} - val loss: {:.3f}".format(epoch,
                                                                        elbo_loss,
                                                                        elbo_val_loss))
        # save model
        if elbo_val_loss > best_loss:
            print('Saving better autoencoder model...')
            print()
            model.save_weights('checkpoint/model')
            best_loss = elbo_val_loss
