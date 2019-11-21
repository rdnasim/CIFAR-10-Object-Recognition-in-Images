"""
Author: Riadul Islam Nasim
File: training a CNN architecture kaggle Cifar-10 project dataset
"""

import keras
import numpy as np


#project modules
from ..import config
from .import my_model, preprocess

# loading data
x_train, y_train = preprocess.load_train_data()
print("train data shape: ", x_train.shape)
print("train data label: ", y_train.shape)


#loading model

model = my_model.get_model()

#compile