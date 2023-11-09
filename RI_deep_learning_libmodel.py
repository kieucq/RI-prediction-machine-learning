#
# NOTE: this lib contains all deep learning models for RI prediction. Note the
# difference between "binary_accuracy" and "accuracy" metrics as mentioned in
# https://keras.io/api/metrics/accuracy_metrics/.
# - binary_accuracy: compare two real numbers to see if they match above a given threshold
# - accuracy: compare two integer numbers to see if they match.
#
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
#
# Build a logistic model with 32/64 nodes. 
# 
def model_logistics32(metric_threshold=0.5):
    model_logistics = keras.Sequential([layers.Dense(32, activation = "relu"),
                                        layers.Dense(64, activation = "relu"),
                                        layers.Dense(1, activation = "sigmoid")])
    #model_logistics.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
    model_logistics.compile(optimizer="rmsprop",loss="binary_crossentropy",
                            metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=metric_threshold)])
    return model_logistics 
#
# Build a logistic model with 64/128 nodes.
#
def model_logistics64(metric_threshold=0.5):
    model_logistics = keras.Sequential([layers.Dense(64, activation = "relu"),
                                        layers.Dense(128, activation = "relu"),
                                        layers.Dense(1, activation = "sigmoid")])
    #model_logistics.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
    model_logistics.compile(optimizer="rmsprop",loss="binary_crossentropy",
                            metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=metric_threshold)])
    return model_logistics
#
# Build an RNN model with 16 nodes
#
def model_RNN16(sequence_length,num_predictors,metric_threshold=0.5):
    inputs = keras.Input(shape=(sequence_length,num_predictors))
    x = layers.SimpleRNN(16, return_sequences=True)(inputs)
    x = layers.SimpleRNN(32, return_sequences=True)(x)
    outputs = layers.SimpleRNN(1,activation = "sigmoid")(x)
    model_RNN = keras.Model(inputs, outputs)
    model_RNN.compile(optimizer="rmsprop",loss="mse",
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=metric_threshold)])
    return model_RNN        
#
# Build an RNN model with 32/64 nodes
#
def model_RNN32(sequence_length,num_predictors,metric_threshold=0.5):
    inputs = keras.Input(shape=(sequence_length,num_predictors))
    x = layers.SimpleRNN(32, return_sequences=True)(inputs)
    x = layers.SimpleRNN(64, return_sequences=True)(x)
    outputs = layers.SimpleRNN(1,activation = "sigmoid")(x)
    model_RNN = keras.Model(inputs, outputs)
    model_RNN.compile(optimizer="rmsprop",loss="mse",
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=metric_threshold)])
    return model_RNN
#
# Build a GRU model with 16 nodes
#
def model_GRU16(sequence_length,num_predictors,metric_threshold=0.5):
    inputs = keras.Input(shape=(sequence_length,num_predictors))
    x = layers.GRU(16, return_sequences=True)(inputs)
    x = layers.GRU(32, return_sequences=True)(x)
    outputs = layers.GRU(1,activation = "sigmoid")(x)
    model_GRU = keras.Model(inputs, outputs)
    model_GRU.compile(optimizer="rmsprop",loss="mse",
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=metric_threshold)])
    return model_GRU    
#
# Build a GRU model with 32/64 nodes
#
def model_GRU32(sequence_length,num_predictors,metric_threshold=0.5):
    inputs = keras.Input(shape=(sequence_length,num_predictors))
    x = layers.GRU(32, return_sequences=True)(inputs)
    x = layers.GRU(64, return_sequences=True)(x)
    outputs = layers.GRU(1,activation = "sigmoid")(x)
    model_GRU = keras.Model(inputs, outputs)
    model_GRU.compile(optimizer="rmsprop",loss="mse",
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=metric_threshold)])
    return model_GRU
