# -*- coding: utf-8 -*-
"""
Source Code for Implementation of Part 1

@author: Shweta Salian, Raghava Mallik
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from numpy import histogramdd
import matplotlib.image as mpimg

# Load the nii image
nii_image_t1 = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_T1w.nii')
nii_image_t2 = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_T2w.nii')

# Load the png image
I1_image = mpimg.imread('Assignment_2_data/Data/I1.png')
J1_image = mpimg.imread('Assignment_2_data/Data/J1.png')

I2_image = mpimg.imread('Assignment_2_data/Data/I2.jpg')
J2_image = mpimg.imread('Assignment_2_data/Data/J2.jpg')

I3_image = mpimg.imread('Assignment_2_data/Data/I3.jpg')
J3_image = mpimg.imread('Assignment_2_data/Data/J3.jpg')

I4_image = mpimg.imread('Assignment_2_data/Data/I4.jpg')
J4_image = mpimg.imread('Assignment_2_data/Data/J4.jpg')

I5_image = mpimg.imread('Assignment_2_data/Data/I5.jpg')
J5_image = mpimg.imread('Assignment_2_data/Data/J5.jpg')

I6_image = mpimg.imread('Assignment_2_data/Data/I6.jpg')
J6_image = mpimg.imread('Assignment_2_data/Data/J6.jpg')

I_J_images = [{1:(I1_image,J1_image)}, {2:(I2_image,J2_image)}, {3:(I3_image,J3_image)}, {4:(I4_image,J4_image)}, {5:(I5_image,J5_image)}, {6:(I6_image,J6_image)}]


# Get the image data from nii_image
t1_img = nii_image_t1.get_fdata()
t2_img = nii_image_t2.get_fdata()


"""
Function to check if two images are of same size
"""
check_shape = lambda x: True if len(x[0].shape)==len(x[1].shape) else False

"""
Function to modify image to make it a 2D sized image
"""
modify_shape = lambda x: x[:,:,0] if len(x.shape)==3 else x


"""
Function to create joint histogram of two images using histogramdd of numpy 
"""
def JoinHist(I, J, bin=10, range=None, density=None, weights=None):
    # Flatten the image for plotting
    I = I.flatten()
    J = J.flatten()
   
    # determine the type of bin is integer or array     
    if type(bin) == list:
        xedges = yedges = np.asarray(bin)
        bins = [xedges, yedges]
    elif type(bin) != list:
        bins = bin
        
    hist, edges = histogramdd([I, J], bins, range, density, weights)
    return hist, edges[0], edges[1]   


"""
Function to plot images I and J and its joint histogram using matplotlib
"""
def plot(I,J,I_title,J_title,histogram,x_label,y_label,histogram_title,log_scale=False):
    # Plot the figure
    fig = plt.figure(figsize=(20, 14))
    
    # Plot I
    fig.add_subplot(1, 3, 1)
    plt.imshow(I)
    plt.title(I_title)
    
    # Plot J
    fig.add_subplot(1, 3, 2)
    plt.imshow(J)
    plt.title(J_title)
    
    # Plotting the joint histogram
    fig.add_subplot(1, 3, 3)
    if log_scale:
        plt.imshow(np.log10(histogram.T + 1), cmap='jet')
    else:
        plt.imshow(histogram, cmap='jet')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(histogram_title)
    
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    
    """
    Part 1 a - Implementation
    """
    
    print(f'The shape of the t1 is {t1_img.shape}')
    print(f'The shape of the t2 is {t2_img.shape}')
    
    # Calculate the joint histogram
    if check_shape((t1_img.flatten(),t2_img.flatten())):        
        histogram, _, _ = JoinHist(t1_img, t2_img, bin=15)
    else:
        raise ValueError('arrays I and J should have the same lengths')
        
    plot(t1_img[:, :, t1_img.shape[2]//2], t2_img[:, :, t2_img.shape[2]//2], 't1', 't2', histogram, 't1 Intensity', 't2 Intensity', 'Joint Histogram of t1 and t2')
    
    
    
    """
    Part 1 b - Implementation
    """
    sum_of_histogram = int(np.sum(histogram))
    print(f'Sum of joint histogram of t1 and t2 is {sum_of_histogram}')
   
    product = 1
    
    for dimension in t1_img.shape:
        product = product * dimension
        
    print(f'Product of each dimension {str(t1_img.shape).replace(",","*")} of t1 is {product}')
    
    if sum_of_histogram == product :
        print('Sum of joint histogram of two images is equal to product of each dimension of the size of an image used')
    
    
    """
    Part 1 c - Implementation
    """   
    
    for image in I_J_images:
        img_no = next(iter(image.keys()))
        img = next(iter(image.values()))
        
        if check_shape(img):
            print(f'Shape of image I{str(img_no)} is {img[0].shape}')
            print(f'Shape of image J{str(img_no)} is {img[1].shape}')
            histogram, _, _ = JoinHist(img[0], img[1], bin=15)
        else:
            # modifying the image by taking slice of it to generate joint histogram
            histogram, _, _ = JoinHist(modify_shape(img[0]), modify_shape(img[1]), bin=15)
            
        # Giving option log_scale as True to use logarithmic scale to visualise joint histogram
        plot(img[0], img[1], 'I{str(img_no)}', 'J{str(img_no)}', histogram, 'I{str(img_no)} Intensity', 'J{str(img_no)} Intensity', 'Joint Histogram of I{str(img_no)} and J{str(img_no)}', log_scale=True)
        
