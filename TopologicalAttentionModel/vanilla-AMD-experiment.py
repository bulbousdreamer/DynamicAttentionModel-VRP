#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:11:24 2023

@author: joseaguilar
"""

# Core python imports
import argparse
import pickle
import time
from time import strftime, gmtime
from datetime import date
import sys
import os
from pathlib import Path


# Typiing imports
from typing import Tuple

# External AI/ML libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# AM-D libraries
from utils import create_data_on_disk
from attention_dynamic_model import AttentionDynamicModel
from reinforce_baseline import RolloutBaseline
from train import train_epoch



###### ARGUMENT PARSING ######
def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = 'Experiment program used for collecting information in training time and performance of AM-D models.',
        description = 'Performs an experiment, and returns a file with data collected.',
        epilog      = 'This is part of my thesis research.'
        )
    parser.add_argument('-v', '--verbose', action='count',   default=0, 
                        help='Verbosity of output')
    parser.add_argument('--check_gpus', action='store_true', default=False, 
                        help='Check if GPUs are being used')
    ### General Combinatorial Problem Settings ###
    parser.add_argument('-n', '--nodes', type=int, default=20, 
                        help='Number of nodes for each problem.')
    ### Validation Data Settings ###
    parser.add_argument('--validation_samples', type=int, default=10_000,
                        help='Number of samples to use for validation dataset')
    parser.add_argument('--validation_seed', type=int, default=42,
                        help='Random seed used for the validation dataset.')
    ### AM-D Parameters ###
    parser.add_argument('--load_model', type=str, default=None,
                        help='Directory of model to load instead of creating a new one.')
    parser.add_argument('-d', '--embedding_dim', type=int,   default=128, 
                        help='The number of dimensions to embed input. NOTE: The embedding dimension MUST be divisible by the number of attention heads (n_heads.)')
    parser.add_argument('-l', '--n_encode_layers', type=int, default=3, 
                        help='Number of encoder layers.')
    parser.add_argument('-a', '--n_heads', type=int,         default=8, 
                        help='Number of attention heads used for both encoder and decoder. NOTE: The number of attention heads must divide the embedding dimension (embedding_dim.)')
    parser.add_argument('--tanh_clipping', type=float,       default=10., 
                        help='Attention clipping value. Avoids "explosion" of values.')
    parser.add_argument('--decode_type', type=str,           default='sampling',
                        help='Sampling mode model to train. Options are "sampling" and "greedy". "Sampling" is encouraged for training, since it allows exploration.')
    ### Rollout Baseline Parameters ###
    parser.add_argument('--baseline_suffix', type=str,       default=None,
                        help='Suffix for checkpoint to be saved as.')
    parser.add_argument('--baseline_from_checkpoint', action='store_true', default=False,
                        help='If True, it will load the checkpoint {path_to_checkpoint}/baseline_checkpoint_epoch_{epoch}_{filename}.h5')
    parser.add_argument('--path_to_baseline_checkpoint', type=str, default='.',
                        help='Root path to baseline checkpoint. Defaults to file\'s home directory.')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warm up epochs for baseline.')
    parser.add_argument('--warmup_data_size', type=int, default=10_000,
                        help='Amount of data to be generated for warmup.')
    parser.add_argument('--warmup_beta', type=float, default=0.9,
                        help='Exponential decay for warmup')
    ### Optimizer Parameters ###
    parser.add_argument('-r', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate for model. Defaults to 1e-4.')
    parser.add_argument('-1', '--beta_1', type=float, default=0.9, 
                        help='Adam\'s exponential decay rate for 1st moment estimate.')
    parser.add_argument('-2', '--beta_2', type=float, default=0.999,
                        help='Adam\'s exponential decay rate for 2nd moment estimate.')
    parser.add_argument('--epsilon', type=float, default=1e-7, 
                        help='Small constant for numerical stability.')
    ### Model Training Parameters ###
    parser.add_argument('-s', '--training_samples', type=int, default=52_000,
                        help='Number of samples for training per epoch.')
    parser.add_argument('-b', '--batches', type=int, default=32,
                        help='Number of batches for training.')
    parser.add_argument('-e', '--training_epochs', type=int, default=50,
                        help='Number of training epochs. NOTE: samples used = training_epochs * training_samples')
    parser.add_argument('--model_from_checkpoint', action='store_true',
                        help='Flag for loading model from a given checkpoint.')
    parser.add_argument('--gradient_norm_clipping', type=float, default=1.0,
                        help='Gradient maximum value at which gradients are clipped and rescaled.')
    parser.add_argument('--model_filename', type=str, default=None,
                        help='Filename where AM-D model is to be loaded from.')
    parser.add_argument('--save_model_customize', type=str, default='',
                        help='Customization token for saving model.')
    parser.add_argument('--stopping_tolerance', type=float, default=None,
                        help='Early stopping epoch tolerance. This is the number of epochs of worsened validation performance to tolerate before calling it "good."')
    # Output of Experiments
    parser.add_argument('--exp_file', type=str, default='.',
                        help='Output of experiment directory.')
    parser.add_argument('--exp_appendix', type=str, default='',
                        help='Experiment appendix string.')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='Flag for saving final trained model. Model name will depend uppon the date, graph nodes, epochs, number of batches, and batch size as well as some custom file argument.')
    ### Misc ###
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Flag used for debugging model. It creates the model, but it will not perform any training.')
    
    
    return parser

def check_cli(cli: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Ensures parameters are valid.
    
    Parameters
    ----------
    cli : argparse.ArgumentParser
        CLI or program.

    Returns
    -------
    cli: argparse.ArgumentParser

    """
    if cli.load_model is not None:
        # This information gets lost when restoring experiment.
        tmp = cli.load_model
        parent_dir = Path(cli.load_model).parent.absolute()
        with open(os.path.join(parent_dir, 'cli.pkl')) as jar:
            cli = pickle.load(jar)
        cli.load_model = tmp # Now we know we restored the experiment.
    if cli.baseline_suffix is None:
        cli.baseline_suffix = 'VRP_Baseline_{}_{}'.format(cli.nodes, strftime("%Y-%m-%d", gmtime()))
    if cli.model_filename is None:
        cli.model_filename = 'VRP_AMD_{}_{}'.format(cli.nodes, strftime("%Y-%m-%d", gmtime()))
        
    return cli
        

###### MODEL SETUP ######
def create_model(cli: argparse.ArgumentParser) -> Tuple[AttentionDynamicModel, RolloutBaseline, keras.optimizers.Optimizer]:
    """Create a ready-to-train model.
    

    Parameters
    ----------
    cli : argparse.ArgumentParser
        Cli arguments. Specifically, these are used:
            * embedding_dim
            * n_encode_layers
            * n_heads
            * tanh_clipping
            * decode_type
            * baseline_suffix
            * baseline_from_checkpoint
            * path_to_baseline_checkpoint
            * warmup_data_size
            * nodes
            * learning_rate
            * beta_1
            * beta_2
            * epsilon

    Returns
    -------
    model_amd : AttentionDynamicModel
        Pre-prepared AM-D model ready for training.
    baseline : RolloutBaseline
        baseline model used for training.
    optimizer : keras.optimizers.Optimizer
        An optimizer for training..

    """
    # Create AM-D model
    if cli.load_model is not None:
        # Note that cli has been restored, so we can trust these parameters.
        model_restore = AttentionDynamicModel(
            embedding_dim  =cli.embedding_dim,
            n_encode_layers=cli.n_encode_layers,
            n_heads        =cli.n_heads,
            tanh_clipping  =cli.tanh_clipping
        )
        model_restore.load_weights(cli.load_model)
        model_amd = model_restore
    else:
        model_amd = AttentionDynamicModel(
            embedding_dim  =cli.embedding_dim,
            n_encode_layers=cli.n_encode_layers,
            n_heads        =cli.n_heads,
            tanh_clipping  =cli.tanh_clipping
        )
    model_amd.set_decode_type(cli.decode_type)
    
    # Create RolloutBaseline
    baseline = RolloutBaseline(
        model             = model_amd,
        filename          = cli.baseline_suffix,
        from_checkpoint   = cli.baseline_from_checkpoint,
        path_to_checkpoint= cli.path_to_baseline_checkpoint,
        wp_n_epochs       = cli.warmup_epochs,
        epoch             = 0,
        num_samples       = cli.warmup_data_size,
        warmup_exp_beta   = cli.warmup_beta,
        embedding_dim     = cli.embedding_dim,
        graph_size        = cli.nodes,
        n_encode_layers   = cli.n_encode_layers,
        n_heads           = cli.n_heads,
        tanh_clipping     = cli.tanh_clipping,
        )
    
    # Create Optimizer
    optimizer = Adam(
        learning_rate=cli.learning_rate,
        beta_1=cli.beta_1,
        beta_2=cli.beta_2,
        epsilon=cli.epsilon,
        name='Adam',
        )
    
    return model_amd, baseline, optimizer
    



if __name__ == '__main__':
    # CLI argument parsing
    parser = parse_arguments()
    cli = parser.parse_args()
    cli = check_cli(cli)
    
    # Initial checks
    if cli.check_gpus:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if cli.debug:
        print("/!\\ Entering Debugging mode. No training will be performed. /!\\")
    
    
    # Create AM-D model
    model, baseline, optimizer = create_model(cli)
    
    # Create a validation dataset
    validation_dataset = create_data_on_disk(
        graph_size =cli.nodes,
        num_samples=cli.validation_samples,
        is_save    =False,
        filename   =None,
        is_return  =True,
        seed       =cli.validation_seed
        )
    
    # Initialize the data collection lists and early stopping counter
    training_time    = []
    performance      = []
    stopping_counter = 0
    
    # If debugging, then, no running of any experiment.
    if cli.debug:
        print("Debugging complete, and model was loaded.")
        sys.exit()
    
    # Main training loop
    for current_epoch in range(cli.training_epochs):
        # Select training data for this epoch
        training_dataset = create_data_on_disk(
            graph_size =cli.nodes,
            num_samples=cli.training_samples,
            is_save    =False,
            filename   =None,
            is_return  =True,
            seed       =None
            )
        # Starts timer
        start_time = time.time()
        # Train!
        train_cost, val_cost = train_epoch(
            optimizer          = optimizer,
            model_tf           = model,
            baseline           = baseline,
            train_dataset      = training_dataset,
            validation_dataset = validation_dataset,
            meta_epoch         = current_epoch,
            batch              = cli.batches,
            val_batch_size     = 1000,
            grad_norm_clipping = cli.gradient_norm_clipping,
            batch_verbose      = 1000,
            cli                = cli)
        # Save time that it took to train.
        training_time.append(time.time() - start_time)

        
        if cli.verbose >= 2:
            print(f"(2v): Epoch took {training_time[-1]} to train.")
        
        # Save the average cost of the models
        performance.append(val_cost.numpy()) # Is this an array or a real number?
        
        # TODO: Permit the storage of the best model so far.
        if cli.stopping_tolerance is not None \
            and current_epoch >= 1 \
            and performance[current_epoch - 1] - performance[current_epoch] > cli.stopping_threshold:
            
            stopping_counter += 1
            if stopping_counter >= cli.stopping_tolerance:
                break
            else:
                stopping_counter = 0
        
    
    # organize collected data and store away.
    if cli.verbose >= 2:
        print(f"(2v): It took {sum(training_time)} to train model")
    if cli.verbose >= 1:
        print("Preparing data collected")
    data_structure = {
        'training_time': training_time,
        'performance': performance,
        'total_training_time': sum(training_time),
        'average_training_time': sum(training_time) / len(training_time),
        'cli': cli
        }
    
    date = date.today().strftime("%b-%d-%Y")
    current_time = time.strftime("%H-%M-%S", time.localtime())
    
    file_name = f'{cli.exp_appendix}_experiment_{date}_{current_time}_nodes_{cli.nodes}.pkl'
    save_file = os.path.join(cli.exp_file, file_name)
    jar = open(save_file, 'wb')                                          
    pickle.dump(data_structure, jar)                                                      
    jar.close()
    if cli.verbose >= 1:
        print(f"Experiment data stored to {save_file}")
        
    if cli.save_model:
        folder_name =  f'{cli.exp_appendix}_experiment_{date}_{current_time}_nodes_{cli.nodes}.ckp'
        model_file = f"AM-D_{date}_{current_time}_nodes_{cli.nodes}_epochs_{cli.training_epochs}_batches_{cli.batches}.ckp"
        model.save_weights(os.path.join(cli.exp_file, folder_name, model_file))
        with open(os.path.join(cli.exp_file, folder_name, 'cli.pkl'), 'wb') as jar:
            jar.dump(cli)
    