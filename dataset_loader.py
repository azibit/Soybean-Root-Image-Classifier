"""Class to load datasets into Fast AI"""

# -*- coding: utf-8 -*-
import time
from random import randint
from yaspin import yaspin
from fastai.vision import *
import datetime

SOYBEAN_ROOT_PATH = 'datasets/SoyBean_Root_Images'

def load_soybean_root_images():
    bs = 8
    transforms = get_transforms(do_flip=True, flip_vert=True, 
                            max_lighting=0.1, max_rotate=359, max_zoom=1.05, max_warp=0.1)
    data = ImageDataBunch.from_folder(SOYBEAN_ROOT_PATH,
                                  valid_pct = 0.2,
                                 size = 512,
                                 bs = bs,
                                  resize_method=ResizeMethod.SQUISH,
                                  ds_tfms=transforms
                                 ).normalize(imagenet_stats)
    
    print('Loading Soybean Root Images ...')
    print(data)
    print("The classes of the images are {0}".format(data.classes))
    
#     #Load the ResNet152 Model
#     model_to_train = load_deep_learning_model(data, models.resnet152)
    
#     #Train the model
#     cycles_in_first_training = 30
#     train_model(model_to_train, cycles_in_first_training)
    
def load_deep_learning_model(data, model):
    """Load the data and the model to use for training"""
    
    print("Load the {0} model...............".format(model))
    model_to_train = cnn_learner(data, model, metrics=[error_rate, accuracy, Precision(), Recall()])
    print("Finished loading the model #########")
    return model_to_train
    
def train_model(model_to_train, cycle):
    currentDT = datetime.datetime.now()
    model_to_train.fit_one_cycle(cycle)
    model_name = str(currentDT)
    model_to_train.save(model_name)
    

load_soybean_root_images()

