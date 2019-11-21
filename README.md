# Soybean-Root-Image-Classifier

# Run this to install all necessary requirements
pip install -r requirements.txt

# Run this script to install the necessary dataset
python util.py 'https://azibit-models-bucket-1.s3.amazonaws.com/SoyBean_Root_Images.zip'
rm 'datasets/SoyBean_Root_Images/susceptible/Susceptible46.jpg'
rm 'datasets/SoyBean_Root_Images/resistant/Resistant19.jpg'

# Run the following to load the images
python dataset_loader.py

# To load a trained model and find new learning rate
# Sample: python dataset_loader.py load_locally_trained_model -option | tee logs/$currentDate.log {model_name=models.resnet152, locally_trained_model_name='2019-11-20 20:27:16.815719',               # dataset_path='datasets/SoyBean_Root_Images'}

python dataset_loader.py -option | tee logs/$currentDate.log load_locally_trained_model {model_name=model_name, locally_trained_model_name=locally_saved_model_name, dataset
_path=path_to_dataset} 

# Retrain trained model
# python dataset_loader.py -option | tee logs/$currentDate.log retrain_trained_model {model_name=models.resnet152, locally_trained_model_name='2019-11-20 20:27:16.815719', dataset
_path='datasets/SoyBean_Root_Images', cycle = 100, slices_min=1e-5, slices_max=1e-4} 


