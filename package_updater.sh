#!/bin/bash
# Script to update all packages

echo "Update all needed packages"
pip3 install -r requirements.txt

#Run the entire code and save the result somewhere
currentDate=`date +"%Y-%m-%d-%T"`
mkdir -p logs

# python dataset_loader.py '{"model_name":"models.resnet152", "locally_trained_model_name":"2019-11-20 20:27:16.815719", "dataset_path":"datasets/SoyBean_Root_Images", "cycle":10, "slices_min":1e-5, "slices_max":1e-4}' -option | tee logs/$currentDate.log

python3 dataset_loader.py -option | tee logs/$currentDate.log
