import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import random
import time
import yaml
import os
       
from tqdm import tqdm

from src.model import VAE, preprocess_images, eval_step


if __name__ == "__main__":
    
    config = yaml.safe_load(open('config.yml'))
    
    # create output dir
    if not os.path.isdir("output"):
        os.makedirs("output")
    
    # model params
    no_classes = config['num_classes']
    input_dims = eval(config['input_shape'])
    latent_dim = config['latent_dim']
    
    # instantiate model class
    model = VAE(input_dims, latent_dim)
    
    # read in weights
    # guild_run_ckpt = os.path.join(path_to_guild_runs, model_run, 'checkpoint')
    guild_run_ckpt = 'checkpoint'     
    latest = tf.train.latest_checkpoint(guild_run_ckpt)
    # load weights
    model.load_weights(latest)
    
    # Load mnist fashion data
    (_, _), (x_test_orig, y_test_orig) = tf.keras.datasets.fashion_mnist.load_data()
    # perform a 7:3 split
    train_len = x_test_orig.shape[0]
    test_len = int(0.3 * train_len)
    x_test, y_test = x_test_orig[-test_len:, :, :], y_test_orig[-test_len:]

    # preprocess
    test_X = preprocess_images(x_test)
    test_y = y_test

    # instantiate test generator
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    # designate batch size
    test_dataset = test_dataset.batch(config['batch_size'])
    test_size = test_X.shape[0]
    # instantiate metric objects
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    # loss function
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # evaluate
    print('Evaluating...')

    # RECONSTRUCTION PLOT
    print("Generating reconstruction sample plot...")
    fig, axs = plt.subplots(nrows=10, ncols=3, sharex=True, sharey=True, figsize=(3,10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # label distribution
    my_label_dist_dict = {key: (np.where(y_test == key)[0], len(np.where(y_test == key)[0])) for key in range(10)}
    # reconstruct and plot
    for key in my_label_dist_dict:
        label_size = my_label_dist_dict[key][1]
        rand_choice = random.randrange(label_size)
        
        label_indices = my_label_dist_dict[key][0]
        rand_idx = label_indices[rand_choice]
        
        sample_orig = x_test[rand_idx]
        sample_proc = test_X[rand_idx]
        sample_proc_rs = sample_proc.reshape((-1, 28, 28, 1))
        
        axs[key,0].imshow(sample_orig)
        axs[key,1].imshow(sample_proc)
        axs[key,2].imshow(model(sample_proc_rs)[0])
    time.sleep(0.5)  
    # save plot
    plot_name = "recon"
    print('Saving reconstruction plot...')
    fig.savefig(os.path.join("output", f"{plot_name}.png"))
    time.sleep(0.5)
    
    # CORRELATION PLOT
    print("Generating latent space correlation plot...")
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title('Latent Space Correlation Matrix')
    ax.set_xlabel('dimension')
    ax.set_ylabel('dimension')
    ax.set_xticks(list(range(latent_dim)))
    ax.set_yticks(list(range(latent_dim)))
    # compute latent space for each label
    label_latent_space = []
    for label in range(9):
        label_indices = my_label_dist_dict[label][0]
        label_samples = test_X[label_indices]
        label_samples = label_samples.reshape((-1, 28, 28, 1))
        z, z_mean, sd = model.encoder(label_samples)
        label_latent_space.append(z.numpy().T)
    
    test_latent_space = np.hstack(label_latent_space)
    test_corr_coef = np.corrcoef(test_latent_space)
    # define plot diplay params
    for i in range(latent_dim):
        for j in range(latent_dim):
            color = 'w'
            if i == j: color = 'purple'
            text = ax.text(j, i, round(test_corr_coef[i, j], 2), ha="center", va="center", color=color)
    
    ax.imshow(test_corr_coef, interpolation="nearest")
    plt.tight_layout()
    time.sleep(0.5)
    print("Saving correlation plot...")
    plt.savefig(os.path.join("output", "latent_corr_mat.png"))
    time.sleep(0.5)
    
    # evaluate loss
    for batch, (images, _) in enumerate(tqdm(test_dataset, total=test_size // config['batch_size'])):
        eval_step(images, model, test_loss)
    time.sleep(0.5)
    print("Test loss: {:.3f}".format(-test_loss.result()))
