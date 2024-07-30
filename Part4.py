# -*- coding: utf-8 -*-
"""
Source Code for Implementation of Part 4  

@author: Shweta Salian, Raghava Mallik
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from scipy.interpolate import interp2d
from Part2 import SSD
import math
from scipy.optimize import minimize


# Loaded images of brain mri
brain_mri_1 = mpimg.imread('Assignment_2_data/Data/BrainMRI_1.jpg')
brain_mri_2 = mpimg.imread('Assignment_2_data/Data/BrainMRI_2.jpg')
brain_mri_3 = mpimg.imread('Assignment_2_data/Data/BrainMRI_3.jpg')
brain_mri_4 = mpimg.imread('Assignment_2_data/Data/BrainMRI_4.jpg')


"""
Function returns a translated image which is translated by vector t=(p,q)
Pre-existing interpolation python function is also used incase p and q values
used are floats
"""
def translation(I,p,q):
    if len(I.shape) ==2:
        I = I[:,:,np.newaxis]
    I = I[:,:,0]
   
    image_x = I.shape[0]
    image_y = I.shape[1]
    x = np.linspace(0,image_x,image_x)
    y = np.linspace(0,image_y,image_y)
    
    interpolator_func = interp2d(y+q,x+p,I,kind='cubic',fill_value=0)
    translated_image = interpolator_func(x,y)
    
    return translated_image;


"""
Function returns a rotated image which is rotated by an angle theta around the
top left of the image using math and numpy
Pre-existing interpolation python function is also used 
"""
def rotate(image,theta):
    if len(image.shape) ==2:    
        image = image[:,:,np.newaxis]
    
    cosine=round(math.cos(theta))
    sine=round(math.sin(theta))
    height=image.shape[0]                                   
    width=image.shape[1]    
                                
    new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))+1
    new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))+1
    
    rotated_image =np.zeros((new_height,new_width,image.shape[2]))
    original_centre_height   = round(((image.shape[0]+1)/2)-1)    
    original_centre_width    = round(((image.shape[1]+1)/2)-1)    
    new_centre_height= round(((new_height+1)/2)-1)        
    new_centre_width= round(((new_width+1)/2)-1)          
    rotation_matrix = np.matrix([[cosine,-sine], [sine,cosine]])
    
    for i in range(height):
        for j in range(width):
            y=image.shape[0]-1-i-original_centre_height                   
            x=image.shape[1]-1-j-original_centre_width
            cord_matr = np.matrix([[y],[x]])
            rotated_cord = np.matmul(rotation_matrix,cord_matr)                                
            new_y = round(rotated_cord[0,0])
            new_x = round(rotated_cord[1,0])
            new_y=new_centre_height-new_y - 1
            new_x=new_centre_width-new_x - 1
            
            if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                rotated_image[new_y,new_x,:]=image[i,j,:]

    return rotated_image


"""
Function for rotation SSD
"""
def rotationSSD(image,theta):
    theta = math.radians(theta)
    cosine=math.cos(theta)
    sine=math.sin(theta)
    height=image.shape[0]                                   
    width=image.shape[1]                                    
    new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))
    new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))
    
    rotated_image=np.zeros((height,width))
    
    original_centre_height   = round(((image.shape[0]+1)/2)-1)    
    original_centre_width    = round(((image.shape[1]+1)/2)-1)    
    new_centre_height= round(((new_height+1)/2)-1)        
    new_centre_width= round(((new_width+1)/2)-1)          
    rotation_matrix = np.matrix([[cosine,-sine], [sine,cosine]])
    
    for i in range(height):
        for j in range(width):
            y=image.shape[0]-1-i-original_centre_height                   
            x=image.shape[1]-1-j-original_centre_width
            cord_matr = np.matrix([[y],[x]])
            rotated_cord = np.matmul(rotation_matrix,cord_matr)                                
            new_y = round(rotated_cord[0,0])
            new_x = round(rotated_cord[1,0])
            new_y=new_centre_height-new_y - 1
            new_x=new_centre_width-new_x - 1
            
            if 0 <= new_x < width and 0 <= new_y < height and new_x>=0 and new_y>=0:
                rotated_image[new_y,new_x]=image[i,j]

    return rotated_image


"""
Function plots a translated image and SSD values while minimizing the SSD
considering only translations
"""
def minimizingSSDTranslation(image1,image2,img1_name,img2_name):
    height=image1.shape[0]                                   
    width=image1.shape[1]
    ssd_data = {}
    for p in range(0,height,5):
        for q in range(0,width,5):
            new_image = translation(image1,p,q)
            ssd_data[p,q] = SSD(new_image,image2)

    ssd_min_key = min(ssd_data, key=ssd_data.get)
    translated_img = translation(image1,ssd_min_key[0],ssd_min_key[1])
    
    plotSSD("Translated", translated_img, image2, img1_name, img2_name, ssd_data)
    


"""
Function plots a rotated image and SSD values while minimizing the SSD
considering only rotations
"""
def minimizingSSDRotation(image1,image2,img1_name,img2_name):
    ssd_data = {}
    for i in range(0,360):
        
        new_image = rotationSSD(image1,i)        
        ssd_data[i] = SSD(new_image,image2)
    
    ssd_min_key = min(ssd_data, key=ssd_data.get)
    rotated_img = rotate(image1,ssd_min_key)
    
    plotSSD("Rotated", rotated_img, image2, img1_name, img2_name, ssd_data)
    

"""
Function plots a rotated and translated image and SSD values while minimizing the SSD
considering both rotations and translations
"""
def gradientDescent(I,J,I_title,J_title):
    height = I.shape[0]
    width = J.shape[1]
    ssd_data = {}

    for p in range(0,180,30):
        for q in range(0,height,80):
            for k in range(0,width,80):
                Image_Translated = translation(I,p,q)
                Image_Rotated = rotationSSD(Image_Translated,p)
                ssd_data[p,q,k] = SSD(Image_Rotated,J)
    
    plt.subplot(2,1,1)
    ssd_min_key = min(ssd_data, key=ssd_data.get)
    Image_Translated = translation(I,ssd_min_key[1],ssd_min_key[2])
    Image_Rotated = rotate(Image_Translated, ssd_min_key[0])
    plt.imshow(Image_Rotated)
    plt.imshow(J, alpha=.5,cmap="jet")
    plt.title(f'Plotting Rotated and Translated {I_title} over {J_title}')
    
    
    plt.tight_layout(h_pad=5, w_pad=5)
    
    plt.subplot(2,1,2)
    plt.title('SSD Curve for each iteration')
    plt.plot(ssd_data.values())
    plt.show()
    
    return ssd_data


"""
Function to plot original and transformed image
"""
def plot(image1, image2, image1_title, image2_title):
    # Plot the figure
    fig = plt.figure(figsize=(20, 14))
    
    # Plotting Original
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title(image1_title)
    
    # Plotting Transformed
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title(image2_title)
    
"""
Function to plot SSD transformed image and SSD curve for each iteration
"""
def plotSSD(transformation, transformed_img, image_2, img1_name, img2_name, ssd_data):
    plt.subplot(2,1,1)
    plt.title(f'Minimum SSD {transformation} Image {img1_name} and {img2_name}')
    plt.imshow(transformed_img)    
    plt.imshow(image_2, alpha=.5, cmap='jet')
    
    plt.tight_layout(h_pad=5, w_pad=5)
    
    plt.subplot(2,1,2)
    plt.title(f'SSD Curve for each iteration {img1_name} and {img2_name}')
    plt.plot(ssd_data.values())
    plt.ylabel('Sum Squared Difference Plot')   
    plt.show()
    
if __name__ == "__main__":

    """
    Part 4 a - Implementation
    """    
    # defining translation vector 
    # translation_vector = (40.5,40.5)
    # # calling function to translate input image
    # translated_image = translation(I3_image, *translation_vector)
    # plot(I3_image, translated_image, 'Original image', 'Translated image')
    
    
    """
    Part 4 b - Implementation
    """
    # minimizingSSDTranslation(brain_mri_1,brain_mri_2,"Brain_mri_1","Brain_mri_2")
    # minimizingSSDTranslation(brain_mri_1,brain_mri_3,"Brain_mri_1","Brain_mri_3")
    # minimizingSSDTranslation(brain_mri_1,brain_mri_4,"Brain_mri_1","Brain_mri_4")

    """
    Part 4 c - Implementation
    """
    # rotated_image = rotate(brain_mri_1,45)
    # plot(brain_mri_1, rotated_image, 'Original image', 'Rotated image')
    
    
    
    """
    Part 4 d - Implementation
    """
    # minimizingSSDRotation(brain_mri_1,brain_mri_2,"Brain_mri_1","Brain_mri_2")
    # minimizingSSDRotation(brain_mri_1,brain_mri_3,"Brain_mri_1","Brain_mri_3")
    # minimizingSSDRotation(brain_mri_1,brain_mri_4,"Brain_mri_1","Brain_mri_4")
    
    
    """
    Part 4 e - Implementation
    """
    ssd_data = gradientDescent(brain_mri_1,brain_mri_2,"Brain_mri_1","Brain_mri_2")
    # gradientDescent(brain_mri_1,brain_mri_3,"Brain_mri_1","Brain_mri_3")
    # gradientDescent(brain_mri_1,brain_mri_4,"Brain_mri_1","Brain_mri_4")
    
    # initial_params = np.zeros(6) 
    # result = minimize(gradientDescent(brain_mri_1,brain_mri_2,"Brain_mri_1","Brain_mri_2"), initial_params, method='L-BFGS-B')
    
    # plt.title('SSD Curve for each iteration optimized')
    # plt.plot(result.values())
    # plt.ylabel('Sum Squared Difference Plot Optimized')   
    # plt.show()

    