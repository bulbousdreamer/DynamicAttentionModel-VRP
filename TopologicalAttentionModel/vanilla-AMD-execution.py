#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:38:38 2023

@author: Jose E. Aguilar Escamilla

This file contains code used for executing a single experiment of the attention
model. I will use this as a simple implementation of the code to compare results
with any changes I made to the model.
"""

from typing import Tuple

import time
from datetime import date
import os
from time import strftime, gmtime
import argparse

import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from attention_dynamic_model import AttentionDynamicModel
from reinforce_baseline import RolloutBaseline
from utils import create_data_on_disk
from train import train_model



def parse_arguments() -> argparse.ArgumentParser:
    """
    Creates the CLI interface for the program

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser(
        prog        = 'Vanilla Dynamic Attention Model: An Example',
        description = 'A quick-to-use AM-D model trainer.',
        epilog      = 'Here be dragons!'
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
    parser.add_argument('-d', '--embedding_dim', type=int,   default=128, 
                        help='The number of dimensions to embed input. NOTE: The embedding dimension MUST be divisible by the number of attention heads (n_heads.)')
    parser.add_argument('-l', '--n_encode_layers', type=int, default=2, 
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
    parser.add_argument('--bseline_from_checkpoint', action='store_true', default=False,
                        help='If True, it will load the checkpoint {path_to_checkpoint}/baseline_checkpoint_epoch_{epoch}_{filename}.h5')
    parser.add_argument('--path_to_baseline_checkpoint', type=str, default='.',
                        help='Root path to baseline checkpoint. Defaults to file\'s home directory.')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warm up epochs for baseline.')
    parser.add_argument('--warmup_data_size', type=int, default=10_000,
                        help='Amount of data to be generated for warmup.')
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
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='Flag for saving final trained model. Model name will depend uppon the date, graph nodes, epochs, number of batches, and batch size as well as some custom file argument')
    parser.add_argument('--save_model_customize', type=str, default='',
                        help='Customization token for saving model.')
    ### Misc ###
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Flag used for debugging model. It creates the model, but it will not perform any training.')
    #parser.add_argument()
    
    return parser

def check_cli(cli: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Parameters
    ----------
    cli : argparse.ArgumentParser
        CLI or program.

    Returns
    -------
    None.

    """
    if cli.baseline_suffix is None:
        cli.baseline_suffix = 'VRP_Baseline_{}_{}'.format(cli.nodes, strftime("%Y-%m-%d", gmtime()))
    if cli.model_filename is None:
        cli.model_filename = 'VRP_AMD_{}_{}'.format(cli.nodes, strftime("%Y-%m-%d", gmtime()))
    return cli
        

def create_model(cli: argparse.ArgumentParser) -> Tuple[AttentionDynamicModel, RolloutBaseline, keras.optimizers.Optimizer]:
    # Create AM-D model
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
        from_checkpoint   = cli.bseline_from_checkpoint,
        path_to_checkpoint= cli.path_to_baseline_checkpoint,
        wp_n_epochs       = cli.warmup_epochs,
        epoch             = 0,
        num_samples       = cli.warmup_data_size,
        embedding_dim     = cli.embedding_dim,
        graph_size        = cli.nodes
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
    




if __name__ == "__main__":
    # Obtain parameters from CLI
    parser = parse_arguments()
    cli = parser.parse_args()
    cli = check_cli(cli)
    # Initial checks
    if cli.check_gpus:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Set up AM-D model using passed arguments
    model, baseline, optimizer = create_model(cli)
    
    # Create validation dataset
    validation_dataset = create_data_on_disk(
        graph_size =cli.nodes,
        num_samples=cli.validation_samples,
        is_save    =False,
        filename   =None,
        is_return  =True,
        seed       =cli.validation_seed
        )
    
    #validation_dataset = None
    
    # Train model
    if not cli.debug:
        # Prepare timer
        start_time = time.time()
        print(start_time)
        
        # Train...
        train_model(optimizer =optimizer,
                    model_tf = model,
                    baseline = baseline,
                    validation_dataset = validation_dataset,
                    samples = cli.training_samples,
                    batch = cli.batches,
                    val_batch_size = 1_000,
                    start_epoch = 0,
                    end_epoch = cli.training_epochs,
                    from_checkpoint = cli.model_from_checkpoint,
                    grad_norm_clipping = cli.gradient_norm_clipping,
                    batch_verbose = cli.verbose,
                    graph_size = cli.nodes,
                    filename = cli.model_filename
                    )
        print(time.time() - start_time )
        
        # Save model
        if cli.save_model:
            date = date.today().strftime("%b-%d-%Y")
            t = time.localtime()
            current_time = time.strftime("%H-%M-%S", t)
            model.save_weights(f"checkpoints/AM-D_{date}_{current_time}_{cli.nodes}_nodes_{cli.training_epochs}_epochs_{cli.batches}_batches.ckp")
        
    #print('end of file')
    #main()
