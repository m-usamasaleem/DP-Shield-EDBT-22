import cv2
import numpy as np
#from matplotlib import pyplot as plt
from app.privacy_methods import util
import random
import sys
from app.privacy_methods import solver


def svd (im_path, i, epsilon):
    '''
    Generate a private image using the Singular Value Decomposition (SVD) mechanism

    Keyword arguments:
    im_path -- The path to the image you want to use as input
    i       -- Algorithm parameter controlling the number of values to sample from decomposed matrix
    epsilon -- Privacy parameter
    '''
    imgmat = cv2.imread(im_path)
    imgmat = cv2.cvtColor(imgmat, cv2.COLOR_BGR2GRAY)

    imgmat_f =  np.float32(imgmat)/255
    U, sigma, V = np.linalg.svd(np.matrix(imgmat_f))

    vector_raw = solver.sampling(i, epsilon)

    ## sort vector by absolute value
    vector = sorted(vector_raw, key=abs, reverse=True)
    
    sigma_p = np.array(sigma[:i])

    ## debugging exceptions 
    if i> len(sigma):
        print ('I='+str(i) + ' and SVD_len=' + str(len(sigma)))
        print (imgmat.shape)

    if i> len(vector):
        print ('I='+str(i) + ' and noise vector_len=' + str(len(vector)))

    ## perturbing the svd values       
    for j in range(i):
        #change = ((-1)**random.randint(0,1))*random.random()*delta
        sigma_p[j] = sigma[j] + vector[j]

    svd_im = np.array(np.matrix(U[:, :i]) * np.diag(sigma_p[:i]) * np.matrix(V[:i, :]))

    svd_im= np.uint8(svd_im*255)
    
    mf_size = 3
    svd_im_median = cv2.medianBlur(svd_im, mf_size)
    svd_im_np = np.array(np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :]))
    svd_im_np = np.uint8(svd_im_np*255)
    
    return svd_im, svd_im_median, svd_im_np