"""Class to load datasets into Fast AI"""

# -*- coding: utf-8 -*-
import time
from random import randint
from yaspin import yaspin
from fastai.vision import *
from fastai import *
from fastai.core import *
from fastai.vision import image as im
import datetime
from PIL import Image   
from sys import argv
import json 
from sklearn.model_selection import StratifiedKFold


SOYBEAN_ROOT_PATH = 'datasets/SoyBean_Root_Images'
LOG_IMAGES_FOLDER = "images_folder/"
RESNET_152 = 'models.resnet152'
RESNET_101 = 'models.resnet101'
VGG19_BN = 'models.vgg19_bn'
LOCALLY_TRAINED_MODEL = '2019-11-30 20:48:02.598479'
TRANSFORMS = get_transforms(do_flip=True, flip_vert=True, 
                            max_lighting=0.1, max_rotate=359, max_zoom=1.05, max_warp=0.1)
IMAGE_SIZE = 512

def get_models(model_name):
    if(model_name == RESNET_152):
        return models.resnet152
    elif(model_name == VGG19_BN):
        return models.vgg19_bn
    elif(model_name == RESNET_101):
        return models.resnet101

def get_dataset(dataset_file):
    print("Getting the dataset we need from the path: {0}".format(dataset_file))
    bs = 8
    data = ImageDataBunch.from_folder(dataset_file,
                                  valid_pct = 0.2,
                                 size = IMAGE_SIZE,
                                 bs = bs,
                                  resize_method=ResizeMethod.SQUISH,
                                  ds_tfms=TRANSFORMS
                                 ).normalize(imagenet_stats)
    print("Completely loaded all dataset images")
    return data

def k_fold_cross_validation(k, dataset_file, model_name, cycle):
    """Perform a k-fold cross-validation"""
    
    # Get the dataset to use.
    data = get_dataset(dataset_file)
    
    print("Transforming the data to dataframe")
    #Get the dataframe from the data
    df = data.to_df()
    
    skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 1)
    acc_val = []
    precision_val = []
    recall_val = []
    
    # Update the model to the new model after every iteratio
    saved_model_new__name = LOCALLY_TRAINED_MODEL
    
    print("Starting cross-validation from here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for train_index, val_index in skf.split(df.index, df['y']):
        print("Cross Validation Starting again &&&")
        data_fold = (ImageList.from_df(df, dataset_file)
                     .split_by_idxs(train_index, val_index)
                     .label_from_df()
                     .transform(TRANSFORMS, size=IMAGE_SIZE)
                     .databunch(num_workers = 2)).normalize(imagenet_stats)
        
        data_fold.batch_size = 16
        
        # Load the model based on the new data
        print("Loading the Deep Learning Model $$$$$$")
        learn = load_deep_learning_model(data_fold, model_name)
        
        print("Load Locally trained Model")
        # Load locally trained model
        # Update the model to the new model after every iteratio
        learn.load(saved_model_new__name)
        
        print("Freeze the first 10 Layers")
        #Freeze the first 10 layers
        learn  = freeze_to(learn, 10)
        
        print("TRAIN! TRAIN!! TRAIN!!!")
        learn.fit_one_cycle(cycle)
        print("Any issues here @@@@@@@@@@@@@@@@@@@@@@")
        x = learn.validate(metrics=[error_rate, accuracy, Precision(), Recall()])
        print("The result is: {0}".format(len(x)))
        last, error, acc, precision, recall = x[0], x[1], x[2], x[3], x[4]
        print("Accuracy: {0}".format(acc))
        print("Loss: {0}".format(error))
        print("Precision: {0}".format(precision))
        print("Recall: {0}".format(recall))
        print("Last: {0}".format(last))
        acc_val.append(acc)
        precision_val.append(precision)
        recall_val.append(recall)
        
        print("SHOW CONFUSION MATRIX")
        # Show the confusion matrix
        show_confusion_matrix(learn)
        
        print("Save Model")
        #Save the model
        saved_model_new__name = current_date_time_as_str()
        learn.save(saved_model_new__name)
        
    mean_accuracy = np.mean(acc_val)
    mean_precision = np.mean(precision_val)
    mean_recall = np.mean(recall_val)
    print("Mean accuracy: {0}".format(mean_accuracy))
    print("Mean precision: {0}".format(mean_precision))
    print("Mean recall: {0}".format(mean_recall))
    
    std_deviation_acc = np.std(acc_val)
    std_deviation_precision = np.std(precision_val)
    std_deviation_recall = np.std(recall_val)
    print("Standard deviation of accuracy: {0}".format(std_deviation_acc))
    print("Standard deviation of precision: {0}".format(std_deviation_precision))
    print("Standard deviation of recall: {0}".format(std_deviation_precision))
    
    print("Completed cross validation at this point ###")
    
    

def load_soybean_root_images():
    data = get_dataset(SOYBEAN_ROOT_PATH)
    print('Loading Soybean Root Images ...')
    print(data)
    print("The classes of the images are {0}".format(data.classes))
    
    #Load the ResNet152 Model
    model_to_train = load_deep_learning_model(data, models.resnet152)
    
    #Train the model
    cycles_in_first_training = 30
    train_model(model_to_train, cycles_in_first_training)

    #Return locally trained model
    #load_locally_trained_model(RESNET_152, LOCALLY_TRAINED_MODEL, SOYBEAN_ROOT_PATH)
    
def load_deep_learning_model(data, model_name):
    """Load the data and the model to use for training"""
    
    print("Load the data and the model to use for training ...")
    model_to_train = cnn_learner(data, get_models(model_name), metrics=[error_rate, accuracy, Precision(), Recall()])

    print("Finished loading the model #########")
    return model_to_train
    
def train_model(model_name, data, cycle):
    """Train a model from base available model"""
    model_to_train = load_deep_learning_model(data, model_name)
    model_to_train.fit_one_cycle(cycle)
    model_name = current_date_time_as_str()
    model_to_train.save(model_name)
    
    return model_to_train
    
def load_locally_trained_model(model_name, locally_trained_model_name, dataset_path):
    """Load a model that has been trained already so it can be used for further experiments"""
    
    print("Load a model that has been trained already so it can be used for further experiments")
    #Load the data to use
    data = get_dataset(dataset_path)
    
    #Load the model to continue training on based on data and model
    model = load_deep_learning_model(data, model_name)
    model.load(locally_trained_model_name)
    
    #plot_learning_rate(model)
    
    return model

def current_date_time_as_str():
    return str(datetime.datetime.now())

def get_images_path():
    return LOG_IMAGES_FOLDER + current_date_time_as_str() + ".png"

def confusion_matrix_images_path():
    return LOG_IMAGES_FOLDER + "conf_matrix_" + current_date_time_as_str() + ".png"

def losses_images_path():
    return LOG_IMAGES_FOLDER + "losses_" + current_date_time_as_str() + ".png"

def plot_learning_rate(model):
    """Plot the learning rate of the model"""
    print("Plot the learning rate of the model")
    model.unfreeze()
    model.lr_find()
    image = model.recorder.plot(return_fig=True)
    
    #Check if the images folder exists, otherwise, create a new folder
    if not os.path.exists(LOG_IMAGES_FOLDER):
        os.makedirs(LOG_IMAGES_FOLDER)
    
    #Save the image in the images folder
    print('Type of image is {0}'.format(type(image)))
    image.savefig(get_images_path())
    print(image)
    
def retrain_trained_model(model_name, locally_trained_model_name, dataset_path, cycle, slices_min, slices_max):
    """Retrain the trained model with new values"""
    
    print("Retrain the trained model with new values")
    # Load the locally trained model
    model = load_locally_trained_model(model_name, locally_trained_model_name, dataset_path)
    
    print("Fitting with new values")
    #Fit with the new provided values
    model.fit_one_cycle(cycle, max_lr=slice(slices_min, slices_max))
    model.save(current_date_time_as_str())
    print("Saved the model")
    
def training_after_initial_unfreeze(value):
    new_value = json.loads(value)
    
    model_name = new_value['model_name']
    locally_trained_model_name = new_value['locally_trained_model_name']
    dataset_path = new_value['dataset_path']
    
    cycle = new_value['cycle']
    slices_min = new_value['slices_min']
    slices_max = new_value['slices_max']
    retrain_trained_model(model_name, locally_trained_model_name, dataset_path, cycle, slices_min, slices_max)

def show_confusion_matrix(model):
    """Show the confusion matrix for the model"""
    preds,y,losses = model.get_preds(with_loss=True)
    interp = ClassificationInterpretation(model, preds, y, losses)
    
    print("Loading the images")
    image = interp.plot_confusion_matrix(return_fig=True)
    image.savefig(confusion_matrix_images_path())
    print("Confusion Matrix image saved")
    
    losses,idxs = interp.top_losses(10)
    data = get_dataset(SOYBEAN_ROOT_PATH)
    print("TOP 10 Images with the highest loss")
    for p in data.valid_ds.x.items[idxs]:
        print(p)
        
def freeze_to(model, layers_to_freeze):
    """Freeze specified number of layers in the model"""
    
    print("Freezing layers now .....")
    model.freeze_to(layers_to_freeze)
    print("Freezing complete")
    
    return model

def initial_training_with_new_model(model_to_train, dataset_path, cycles_to_train):
    # Get the dataset we need
    data = get_dataset(dataset_path)
    
    model = train_model(model_to_train, data, cycles_to_train)
    plot_learning_rate(model)

# training_after_initial_unfreeze(argv[1])
# show_confusion_matrix()
# k_fold_cross_validation(5, SOYBEAN_ROOT_PATH, VGG19_BN, 50)

initial_training_with_new_model(RESNET_101, SOYBEAN_ROOT_PATH, 150)
# retrain_trained_model(VGG19_BN, LOCALLY_TRAINED_MODEL, SOYBEAN_ROOT_PATH, 100, 1e-6, 1e-3)