import os
import cv2
import glob
import time
import math
import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def get_top_rgbs_kmeans(image, m_param, k):
  im = image.copy()
  
  # width = int(im.shape[1] * .5)
  # height = int(im.shape[0] * .5)

  # im = cv2.resize(im, (width, height))

  kmeans_highest_RGBs = []

  pixel_values = im.reshape((-1, 3))
  pixel_values = np.float32(pixel_values)
  hist_entire_im = cv2.calcHist([image], [0], None, [256], [0, 256])

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
  _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  centers = np.uint8(centers)
  labels = labels.flatten()

  for label in range(0, k):
      a = pixel_values[labels==label].mean(axis=1).astype(np.uint8)
      c = np.bincount(a)

      rgb = np.argmax(c)

      rgb_freq = int(hist_entire_im[rgb])
      if rgb_freq > m_param:
        kmeans_highest_RGBs.append( (rgb, rgb_freq) )
  
  
  return labels, centers, kmeans_highest_RGBs

'''
Calculate RGB budgets

'''
def get_rgb_budgets(t_rgbs, epsilon, total_rgb_freq):
  '''
  Divides the epsilon value into k parts and allocates those parts to certain RGBs
  
  t_rgbs: the chosen RGBs and their counts tuple: (rgb, count)
  epsilon: the entire privacy budget
  total_rgb_freq: a list of tuples. tuples in the form: (Intensity value (theta), count of theta in entire VE)
  
  returns: a list of tuples. tuples in the form: (intensity value, count of theta in entire VE, assigned epsilon))
  '''
  rgb_budgets = []

  for rgb_freq in t_rgbs:
      rgb_budget = ( rgb_freq[0], rgb_freq[1], (rgb_freq[1] / total_rgb_freq) * epsilon )
      rgb_budgets.append(rgb_budget)
      
  return rgb_budgets

'''
Calculate optimal RGBs to sample

'''
def compute_optimal_xis(rgbs_and_budgets, m=1, debug=False):
  '''
  Solve finding the optimal x_i for each RGB using binary search
  
  rgbs_and_budgets: list of tuples in the form: (intensity value, count of theta in entire VE, assigned epsilon))
  return: a list of tuples in the form: (intensity value, count of theta in entire VE, assigned epsilon, optimal x_i)
  '''
  
  optimal_xis_for_rgbs = []
  
  TOTAL_global = 0 # FOR TESTING
  TOTAL_local = 0 # FOR TESTING
  
  # iterate every RGB that requires privacy
  for row in rgbs_and_budgets:
      rgb = row[0]
      rgb_count_entire_VE = row[1]
      rgb_allocated_privacy = row[2]
      
      TOTAL_global += math.e**rgb_allocated_privacy ## FOR TESTING
      
      c_i = rgb_count_entire_VE
      range_of_xi = np.arange(1, c_i) # possible xi in interval [1, c_i - 1]
      
      # combination function
      def n_choose_r(n, r):
          assert n>=r, f'n < r (n={n}, r={r})'
          numer = math.factorial(n)
          denom = math.factorial(r)*math.factorial(n-r)

          return int(numer//denom)
      
      def compute_proportion(c_i, x_i):
          return n_choose_r(c_i, x_i) / n_choose_r(c_i - m, x_i)
      

      # search for xi that maximizes privacy use while staying within budget
      converged = False
      lower = 1
      upper = range_of_xi[-1]
      last_valid = 0
      
      # binary search to find optimal x_i
      while lower <= upper:                
          mid = (lower+upper) // 2

          # we have to protect m pixels, so # sampled pixels must be < (# in RGB - m)
          if mid >= c_i-m:
              upper = mid-1
              continue

          proportion = compute_proportion(c_i, mid)
          
          # this x is valid, ignore everything lower
          if proportion <= math.e**rgb_allocated_privacy:
              last_valid = mid
              lower = mid + 1
              
          # this x is invalid, ignore everything higher
          elif proportion > math.e**rgb_allocated_privacy:
              upper = mid-1


      # for debugging. determines if the actual compromised privacy is less than allocated privacy
      privacy_satisfied = compute_proportion(c_i, last_valid) <= math.e**rgb_allocated_privacy


      if debug:
          TOTAL_local += compute_proportion(c_i, last_valid)
          print(f'{last_valid} ({c_i} - {round(100*last_valid/c_i,4)}%) --- {math.e**rgb_allocated_privacy} --- {compute_proportion(c_i, last_valid)} --- {privacy_satisfied}')
      
      # a tuple of (RGB value, RGB count in entire VE, Allocated privacy budget, optimal x_i value)
      rgb_data = (rgb, rgb_count_entire_VE, rgb_allocated_privacy, last_valid)
      optimal_xis_for_rgbs.append(rgb_data)
  
  if debug:
      test = TOTAL_local <= TOTAL_global
      print(f'\nTotal privacy: {TOTAL_global} --- Local privacy: {TOTAL_local} --- Local <= Global? {test}')
  return optimal_xis_for_rgbs


'''
Sample x_i pixels for each RGB_i

'''
def sample_pixels(image, optimal_xis):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    new_image = np.ones( (image.shape[0], image.shape[1]) ) * -1
#     new_image = np.ones( (image.shape[0], image.shape[1]) ) * 0
    
    for rgb in optimal_xis:
        intensity = rgb[0]
        num_px_to_sample = rgb[3]
        
        sampling_candidates = np.where(image == intensity)
        num_candidates = rgb[1]
        
        to_sample = np.random.choice(num_candidates, num_px_to_sample, replace=False)
#         to_sample = np.random.choice(num_candidates, num_candidates, replace=False)
        
        new_image[ sampling_candidates[0][to_sample], sampling_candidates[1][to_sample] ] = intensity
        
    return new_image


'''
Sample x_i pixels for each RGB_i

'''
def interpolate_image_scipy(sampled_image):
  if sampled_image.max()==-1.0:
    print('Sampled 0 pixels')
    return sampled_image

  array = sampled_image.copy()
  array[array==-1] = np.nan
  
  # if epsilon is very low and/or k is very high, the number of sampled pixels can be very low
  # depending on the number of sampled pixels and their position, convex hull may not work properly
  # if this is the can, we use nearest interpolation
  try:
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    
    array = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                  method='linear')
  except:
    pass
  
  x = np.arange(0, array.shape[1])
  y = np.arange(0, array.shape[0])
  #mask invalid values
  array = np.ma.masked_invalid(array)
  xx, yy = np.meshgrid(x, y)
  #get only the valid values
  x1 = xx[~array.mask]
  y1 = yy[~array.mask]
  newarr = array[~array.mask]

  array = interpolate.griddata((x1, y1), newarr.ravel(),
                            (xx, yy),
                                method='nearest')
  
  return array

def run_samplingdp_on_image(im_path, k=15, epsilon=1.6, m_param=1, verbose=False):
  '''
  Generate a private image using the SamplingDP mechanism

  Keyword arguments:
  im_path -- The path to the image you want to use as input
  k       -- Algorithm parameter that controls the number of clusters to use for private sampling
  epsilon -- Privacy budget
  m_param -- The number of pixels to be protected by the mechanism
  verbose -- Toggles printing of the steps of mechanism
  '''
  im = cv2.imread(im_path)

  if verbose: print('KMeans clustering...')
  labels, centers, t_rgbs = get_top_rgbs_kmeans(im, m_param, k)

  # calculate total rgbs that need budget
  total_rgb_freq = np.array([i[1] for i in t_rgbs]).sum()

  if verbose: print('Calculating rgb budgets...')
  rgbs_and_budgets = get_rgb_budgets(t_rgbs, epsilon, total_rgb_freq)

  if verbose: print('Calculating optimal x_i...')
  optimal_xis = compute_optimal_xis(rgbs_and_budgets, m=m_param)

  if verbose: print('Sampling x_i pixels for each rgb...')
  sampled_image = sample_pixels(im, optimal_xis)

  if verbose: print('Interpolating pixels in sampled image...')
#     private_image = interpolate_image(sampled_image)
  private_image = interpolate_image_scipy(sampled_image)
  
  return (im, sampled_image, private_image)