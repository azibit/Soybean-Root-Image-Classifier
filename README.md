# Soybean-Root-Image-Classifier

# Run this to install all necessary requirements
pip install -r requirements.txt

# Run this script to install the necessary dataset
python util.py 'https://azibit-models-bucket-1.s3.amazonaws.com/SoyBean_Root_Images.zip'
rm 'datasets/SoyBean_Root_Images/susceptible/Susceptible46.jpg'
rm 'datasets/SoyBean_Root_Images/resistant/Resistant19.jpg'

# Run the following to load the images
python dataset_loader.py

## REMEMBER TO ALWAYS SAVE YOUR PROGRESS WHEN YOU CALL A PYTHON SCRIPT
# To load a trained model and find new learning rate
# python dataset_loader.py '{"model_name":"models.resnet152", "locally_trained_model_name":"2019-11-20 20:27:16.815719", "dataset_path":"datasets/SoyBean_Root_Images"}' -option | tee logs/$currentDate.log 

##
# Retrain trained model
# python dataset_loader.py '{"model_name":"models.resnet152", "locally_trained_model_name":"2019-11-20 20:27:16.815719", "dataset_path":"datasets/SoyBean_Root_Images", "cycle":10, "slices_min":1e-5, "slices_max":1e-4}' -option | tee logs/$currentDate.log