import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import os
import sys

class TransformerEncoder(layers.layer):
    """
    implements a single transformer encoder block
    """
    def __init__(self,head_size,num_heads,ff_dim,dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.head_size=head_size
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.dropout=dropout
        self.attention=layers.MultiHeadAttention(
            key_dim=head_size,num_heads=num_heads,dropout=dropout
        )
        self.dropout_layer=layers.Dropout(dropout)
        self.layer_norm1=layers.LayerNormalization(epsilon=1e-6)
        self.ffn=keras.Sequential(
            [
                layers.Conv1D(filters=ff_dim,kernel_size=1,activation='relu'),
                layers.Dropout(dropout),
                layers.Conv1D(filters=inputs.shape[-1],kernel_size=1),
            ]
        )
        self.layer_norm2=layers.LayerNormalization(epsilon=1e-6)

    def call(self,inputs):
        #attention mechanism
        x=self.attention(inputs,inputs)
        x=self.dropout_layer(x)
        x=self.layer_norm1(inputs+x)
        #feedforward neural network
        x=self.ffn(x)
        x=self.dropout_layer(x)
        x=self.layer_norm2(x+inputs)
        return x

class TransformerModel(keras.Model):
    def __init__(self,input_shape,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,dropout=0,mlp_dropout=0,**kwargs):
        super().__init__(**kwargs)
        self.input_shape=input_shape
        self.head_size=head_size
        self.num_heads=num_heads
        self.head_size=head_size
        self.num_transformer_blocks=num_transformer_blocks
        self.ff_dim=ff_dim
        self.mlp_units=mlp_units
        self.mlp_dropout=mlp_dropout
        
        self.input_layer=layers.Input(shape=input_shape)
        self.reshape_layer=layers.Resahpe((input_shape,1))

        self.encoder_blocks=[
            TransformerEncoder(head_size,num_heads,ff_dim,dropout) for _ in range(num_transformer_blocks)
        ]

        # Flatten and MLP layers
        self.flatten_layer = layers.Flatten()
        self.mlp_layers = [layers.Dense(units=dim, activation="relu") for dim in mlp_units]
        self.dropout_layer = layers.Dropout(mlp_dropout)

        # Output layer for binary classification
        self.output_layer = layers.Dense(units=1, activation="sigmoid")
    
    def call(self, inputs):
        x = self.reshape_layer(inputs)
        for encoder in self.encoder_blocks:
            x = encoder(x)
        x = self.flatten_layer(x) 
        for mlp in self.mlp_layers:
            x = mlp(x)
            x = self.dropout_layer(x)
        return self.output_layer(x)
