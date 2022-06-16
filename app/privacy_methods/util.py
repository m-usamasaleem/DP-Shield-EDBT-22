# import the necessary packages
from skimage.metrics import structural_similarity as ssim
#import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
    error = np.mean((imageA - imageB)**2)
    return error

def compare_images(imageA, imageB):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)


