"""Utility Module to run all needed methods at all times"""
import wget
from sys import argv
import os
import zipfile
from pathlib import Path

LOCAL_DIR_LOCATION = 'datasets/'
SOYBEAN_ROOT_IMAGES = 'https://azibit-models-bucket-1.s3.amazonaws.com/SoyBean_Root_Images.zip'

def download_url(url=SOYBEAN_ROOT_IMAGES):
    """Download files from a given URL to the specified local file location"""
    
    local_dir_location = 'datasets/'
    local_dir_exists = os.path.isdir(local_dir_location)
    
    if(local_dir_exists):
        #Check if the local directory already exists
        print('Local directory {0} exists'.format(local_dir_location))
        downloaded_file_exists = os.path.exists(local_dir_location + os.path.basename(url))
        
        #Check if the file to download already exists
        if(downloaded_file_exists):
            print('File to download already exists')
        else:
            print('Downloading the file now .....................')
            wget.download(url, local_dir_location)
            print("All done now ######################################")
            
    else:
        #Create the directory and download the needed files
        print('Local directory {0} non existent'.format(local_dir_location))
        os.makedirs(local_dir_location)
        print('Now created {0}'.format(local_dir_location))
        print('Downloading images from', url, 'to', local_dir_location)
        wget.download(url, local_dir_location)
        print("All done now ######################################")
    
    unzip_file(local_dir_location + os.path.basename(url)) 
        
        
def unzip_file(zipped_file_path):
    unzipped_folder = zipped_file_path.split("/")[-1].split(".")[0] 
    
    unzipped_folder_path = LOCAL_DIR_LOCATION + unzipped_folder + '/'
    unzipped_folder_exists = os.path.isdir(unzipped_folder_path)
    
    if(unzipped_folder_exists):
        print('File already unzipped &&&&&&&&&&&&&&&&&&&&&&&&&')
    else:
        print('Unzipped file does not exist. #################')
        with zipfile.ZipFile(zipped_file_path, 'r') as zip_ref:
            print('Extracting now #####################')
            zip_ref.extractall(LOCAL_DIR_LOCATION)  
            print('Completely extracted #####################')
        
 
        
url = argv[1]
download_url(url)

