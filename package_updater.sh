#!/bin/bash
# Script to update all packages

echo "Update all needed packages"
# pip install -r requirements.txt

#Run the entire code and save the result somewhere
currentDate=`date +"%Y-%m-%d-%T"`
mkdir -p logs

python dataset_loader.py -option | tee logs/$currentDate.log load_locally_trained_model {model_name=models.resnet152, locally_trained_model_name='2019-11-20 20:27:16.815719', dataset_path='datasets/SoyBean_Root_Images'}