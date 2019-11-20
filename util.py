"""Utility Module to run all needed methods at all times"""
import wget

def download_url(url, local_file_location):
    """Download files from a given URL to the specified local file location"""
    
    print('Downloading images from', url, 'to', local_file_location)
    wget.download(url, local_file_location)