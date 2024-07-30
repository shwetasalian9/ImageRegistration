# -*- coding: utf-8 -*-
"""
Source Code for Implementation of Part 2

@author: Shweta Salian, Raghava Mallik
"""

import numpy as np
from Part1 import I_J_images, check_shape


"""
Function to calculate Sum of Squared Differences between images I and J using the formula
"""
def SSD(I,J):    
    return np.sum(np.power(I-J,2))



"""
Function to calculate the pearson correlation coefficient between images I and J using the formula
"""
def CORR(I,J):
    return np.cov(I.flatten(),J.flatten())/(np.std(I)*np.std(J))



"""
1) Create 2D histogram of flatten versions of both images
2) Create joint probability of I and J by normalizing the histogram
3) Create marginal probability distribution of I by sum of joint probability along axis=1
4) Create marginal probability distribution of J by sum of joint probability along axis=0
5) Calculate the product of the marginal probability distributions 
6) Calculate mutual information between I and J using the formula
"""
def MI(I,J):
    hist, x_edges, y_edges = np.histogram2d(I.flatten(),J.flatten(),bins=256)
    joint_prob_xy = hist / float(np.sum(hist))
    marginal_prob_x = np.sum(joint_prob_xy, axis=1)
    marginal_prob_y = np.sum(joint_prob_xy, axis=0)
    prod_marginal_prob = marginal_prob_x[:, None] * marginal_prob_y[None, :]
    flag = joint_prob_xy > 0
    return np.sum(joint_prob_xy[flag] * np.log(joint_prob_xy[flag] / prod_marginal_prob[flag]))




if __name__ == "__main__":
    
    """
    Part 2 a - Implementation
    """
    print("Calculating the sum squared difference between two images I and J:")
    for image in I_J_images:
        img_no = next(iter(image.keys()))
        img = next(iter(image.values()))
        if check_shape(img):
            print(f'SSD of Image I{str(img_no)} and Image J{str(img_no)} is {SSD(*img)}')
        else:
            print(f'Images I{str(img_no)} and J{str(img_no)} are not of same shape')
        
    print("\n")
    
    
    
    """
    Part 2 b - Implementation
    """
    print("Calculating the pearson correlation coefficient between two images I and J:")
    for image in I_J_images:
        img_no = next(iter(image.keys()))
        img = next(iter(image.values()))
        if check_shape(img):
            print(f'CORR of Image I{str(img_no)} and Image J{str(img_no)} is {CORR(*img)}')
        else:
            print(f'Images I{str(img_no)} and J{str(img_no)} are not of same shape')
            
            
    print("\n")
        
    
    
    """
    Part 2 c - Implementation
    """
    print("Calculating the mutual information between two images I and J:")
    for image in I_J_images:
        img_no = next(iter(image.keys()))
        img = next(iter(image.values()))
        if check_shape(img):
            print(f'MI of Image I{str(img_no)} and Image J{str(img_no)} is {MI(*img)}')
        else:
            print(f'Images I{str(img_no)} and J{str(img_no)} are not of same shape')