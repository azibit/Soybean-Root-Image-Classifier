# Soybean-Root-Image-Classifier

# Run this to install all necessary requirements
pip install -r requirements.txt

# Run this script to install the necessary dataset
python util.py 'https://azibit-models-bucket-1.s3.amazonaws.com/SoyBean_Root_Images.zip'
rm 'datasets/SoyBean_Root_Images/susceptible/Susceptible46.jpg'
rm 'datasets/SoyBean_Root_Images/resistant/Resistant19.jpg'

# Run the following to load the images
python dataset_loader.py



