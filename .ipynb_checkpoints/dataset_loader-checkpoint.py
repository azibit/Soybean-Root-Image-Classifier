"""Class to load datasets into Fast AI"""

from fastai.vision import *
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
    print(data.shape)
    
    
load_soybean_root_images()