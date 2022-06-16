import cv2
import numpy as np

def apply_snow(im_path, p=0.01):
    '''
    Generate a private image using the Snow mechanism

    Keyword arguments:
    im_path -- The path to the image you want to use as input
    p       -- Proportion of pixels to "flip" to Snow (Intensity=127). delta = 1-p
    '''
    im = cv2.imread(im_path)
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    f = im.flatten()

    num_pixels = im.shape[0]*im.shape[1]
    num_pixels_to_select = int(p * num_pixels)

    im_indices = np.arange(num_pixels)
    indices_to_privatize = np.random.choice(im_indices, num_pixels_to_select, replace=False)

    f[indices_to_privatize] = 127

    private_image = f.reshape((im.shape[0], im.shape[1]))  
    return private_image

def apply_snow_2_im(im, p=0.01):
    '''
    Generate a private image using the Snow mechanism

    Keyword arguments:
    im -- The image you want to use as input
    p       -- Proportion of pixels to "flip" to Snow (Intensity=127). delta = 1-p
    '''
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    f = im.flatten()

    num_pixels = im.shape[0]*im.shape[1]
    num_pixels_to_select = int(p * num_pixels)

    im_indices = np.arange(num_pixels)
    indices_to_privatize = np.random.choice(im_indices, num_pixels_to_select, replace=False)

    f[indices_to_privatize] = 127

    private_image = f.reshape((im.shape[0], im.shape[1]))  
    return private_image