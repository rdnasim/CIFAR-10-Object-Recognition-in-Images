"""
Author: Riadul Islam Nasim
File: preprocessing kaggle Cifar-10 project train and test image
"""

import numpy as np
import cv2, os
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

#project package
from .. import config


X_data = np.ndarray((config.nb_train_samples,
            config.img_size, config.img_shape, config.img_channel), dtype = np.float32)
def load_train_data():
    train_data_dir = os.path.join(config.dataset_path(), "train")
    #print(os.listdir(train_data_dir))
    train_images = sorted(os.listdir(train_data_dir),
            key = lambda x: int(x.split(".")[0]))
    
    #print(train_images)
    train_images = [os.path.join(train_data_dir, img_path)
                     for img_path in train_images]
    
    train_labels_df = pd.read_csv(os.path.join(config.dataset_path(), 
                    "trainLabels.csv"))

    train_labels = train_labels_df["label"].values

    encoder = LabelBinarizer()
    train_labels = encoder.fit_transform(train_labels)
    print(train_labels[100])


load_train_data()


