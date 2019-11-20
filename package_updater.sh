#!/bin/bash
# Script to update all packages

echo "Update all needed packages"
pip install -r requirements.txt

#Run the entire code and save the result somewhere
currentDate=`date +"%Y-%m-%d-%T"`
mkdir -p logs
python dataset_loader.py -option | tee logs/$currentDate.log