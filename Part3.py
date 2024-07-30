# -*- coding: utf-8 -*-
"""
Source Code for Implementation of Part 3

@author: Shweta Salian, Raghava Mallik
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the grid dimensions as shown in figure 1 a 

# Scale of x axis is from 0 to 20
x_min, x_max = 0, 20

# Scale of y axis is from 0 to 20
y_min, y_max = 0, 20

# Scale of z axis is from 0 to 4
z_min, z_max = 0, 4

M1 = np.array([[0.9045, -0.3847, -0.1840, 10.0000],
               [0.2939,  0.8750, -0.3847, 10.0000],
               [0.3090,  0.2939,  0.9045, 10.0000],
               [0.0000,  0.0000,  0.0000,  1.0000]])


M2 = np.array([[-0.0000, -0.2598, 0.1500, -3.0000],
               [0.0000,  -0.1500, -0.2598, 1.5000],
               [0.3000,  -0.0000,  0.0000 , 0],
               [0.0000,  0.0000,  0.0000,  1.0000]])


M3 = np.array([[0.7182, -1.3727, -0.5660, 1.8115],
               [-1.9236,  -4.6556, -2.5512, 0.2873],
               [-0.6426,  -1.7985,  -1.6285 , 0.7404],
               [0,  0,  0,  1.0000]])





"""
Function to plot 3D grid of evenly spaced points using scatter from matplotlib
"""
def plot_grid(X, Y, Z, title, X_transform=None, Y_transform=None, Z_transform=None, transform=False):
    # Plot the grid
    fig = plt.figure()
    fig = fig.add_subplot(111, projection='3d')
    if transform:
        fig.scatter(X, Y, Z, c='b', marker='o', alpha=0.2)
        fig.scatter(X_transform, Y_transform, Z_transform, c='r', marker='o')
    else:
        fig.scatter(X, Y, Z, c='b', marker='o')
        

    # setting aspect as equal for all axis to have relative scaling of points
    fig.set_aspect('equal')

    # this code is to have common start point of '0' for both x and y axis
    plt.xlim([x_max, x_min])

    # this code marks z axis with 2, 4 scales to match it with figure 1a
    fig.set_zticks([2, 4])

    # Set title of the plot and labels of axis
    fig.set_xlabel('X')
    fig.set_ylabel('Y')
    fig.set_zlabel('Z')
    fig.set_title(title)

    # Display the plot
    plt.show()
    

"""
Function to translate X, Y and Z axis array points by given vector
"""
def translate(X_array, Y_array, Z_array, translation_vector):
    return X_array + translation_vector[0], Y_array + translation_vector[1], Z_array + translation_vector[2]



"""
Function to rotate X, Y and Z axis array points by given angle and around
given axis
"""
def rotate(X_array, Y_array, Z_array, axis, angle):
    # Apply rotation around the desired axis
    angle_rad = np.radians(angle)  # Convert angle to radians
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    if axis.__eq__("x-axis"):
        # Apply the rotation matrix to the grid coordinates to rotate around x-axis
        Y_array_rot = Y_array * cos_angle - Z_array * sin_angle
        Z_array_rot = Y_array * sin_angle + Z_array * cos_angle
        return X_array, Y_array_rot, Z_array_rot
    elif axis.__eq__("y-axis"):
        # Apply the rotation matrix to the grid coordinates to rotate around y-axis
        X_array_rot = X_array * cos_angle + Z_array * sin_angle
        Z_array_rot = X_array * sin_angle - Z_array * cos_angle
        return X_array_rot, Y_array, Z_array_rot
    elif axis.__eq__("z-axis"):
        # Apply the rotation matrix to the grid coordinates to rotate around z-axis
        X_array_rot = X_array * cos_angle - Y_array * sin_angle
        Y_array_rot = X_array * sin_angle + Y_array * cos_angle
        return X_array_rot, Y_array_rot, Z_array

    

"""
Function to rigid transform the 3D points cloud to rotate it around all 3 axis
and translate by vector t=(p,q,r) 
Implemented using rotate and translate functions
"""
def rigid_transform(theta, omega, phi, p, q, r):
    rigid_homogenous_cords = dict()
    X, Y_rot, Z_rot = rotate(X_array, Y_array, Z_array, "x-axis", theta) 
    rigid_homogenous_cords["x-axis-rigid"] = (f'Rotation of angle {theta} around the x-axis', X, Y_rot, Z_rot)
    
    X_rot, Y, Z_rot = rotate(X_array, Y_array, Z_array, "y-axis", omega) 
    rigid_homogenous_cords["y-axis-rigid"] = (f'Rotation of angle {omega} around the y-axis', X_rot, Y, Z_rot)
    
    X_rot, Y_rot, Z = rotate(X_array, Y_array, Z_array, "z-axis", phi) 
    rigid_homogenous_cords["z-axis-rigid"] = (f'Rotation of angle {phi} around the z-axis', X_rot, Y_rot, Z)
    
    translation_vector = np.array([p, q, r])
    X_translate, Y_translate, Z_translate = translate(X_array, Y_array, Z_array,translation_vector)
    rigid_homogenous_cords["translate"] = (f'translation of vector t=({p},{q},{r})', X_translate, Y_translate, Z_translate)
    
    return rigid_homogenous_cords



"""
Function to affline transform the 3D points cloud to rotate it around all 3 axis 
and translate by vector t=(p,q,r) and also scaling it by factor of s
Implemented by multiplying output from rotate and transform functions by s
"""
def affine_transform(s, theta, omega, phi, p, q, r):
    affline_homogenous_cords = dict()
    X, Y_rot, Z_rot = rotate(X_array, Y_array, Z_array, "x-axis", theta) 
    affline_homogenous_cords["x-axis-affline"] = (f'Rotation of angle {theta} around the x-axis and scaling by {s} factor', X*s, Y_rot*s, Z_rot*s)
    
    X_rot, Y, Z_rot = rotate(X_array, Y_array, Z_array, "y-axis", omega) 
    affline_homogenous_cords["y-axis-rot"] = (f'Rotation of angle {omega} around the y-axis and scaling by {s} factor', X_rot*s, Y*s, Z_rot*s)
    
    X_rot, Y_rot, Z = rotate(X_array, Y_array, Z_array, "z-axis", phi) 
    affline_homogenous_cords["z-axis-rot"] = (f'Rotation of angle {phi} around the z-axis and scaling by {s} factor', X_rot*s, Y_rot*s, Z*s)
    
    translation_vector = np.array([p, q, r])
    X_translate, Y_translate, Z_translate = translate(X_array, Y_array, Z_array,translation_vector)
    affline_homogenous_cords["translate"] = (f'translation of vector t=({p},{q},{r}) and scaling by {s} factor', X_translate*s, Y_translate*s, Z_translate*s)
    
    return affline_homogenous_cords


if __name__ == "__main__":

   
    """
    Part 3 a - Implementation
    Generate the grid of evenly spaced points
    """
    x_evenly_spaced_nums = np.linspace(x_min, x_max, 21)
    y_evenly_spaced_nums = np.linspace(y_min, y_max, 21)
    z_evenly_spaced_nums = np.linspace(z_min, z_max, 5)
    
    # returns 3-d array of x,y and z vectors
    X_array, Y_array, Z_array = np.meshgrid(x_evenly_spaced_nums, y_evenly_spaced_nums, z_evenly_spaced_nums)
    
    plot_grid(X_array, Y_array, Z_array, '3D Grid of evenly spaced points')
    
    
    
    """
    Part 3 b - Implementation
    Rigid transform the 3D points cloud to rotate it around all 3 axis
    and translate by vector t=(p,q,r) 
    """
    rigid_homogenous_cords = rigid_transform(30,30,10,4,3,2)
    for axis,coordinates in rigid_homogenous_cords.items():
        plot_grid(X_array, Y_array, Z_array,*coordinates,transform=True)
        
    
    """
    Part 3 c - Implementation
    Affline transform the 3D points cloud to perform same operations like Part 3b
    but scale by factor s
    """
    affline_homogenous_cords = affine_transform(0.5,30,30,10,4,3,2)
    for axis,coordinates in affline_homogenous_cords.items():
        plot_grid(X_array, Y_array, Z_array,*coordinates,transform=True)
        
    
    """
    Part 3 d - Implementation
    Decomposing each matrix to determine if each matrix is rigid or affline
    """
    U1,S1,Vt1 = np.linalg.svd(M1)
    U2,S2,Vt2= np.linalg.svd(M2)
    U3,S3,Vt3 = np.linalg.svd(M3)

    print("Decomposition matrix for M1:")
    print("U:")
    print(U1)
    print("S:")
    print(S1)
    print("Vt:")
    print(Vt1)
    print("Decomposition matrix for M2:")
    print("U:")
    print(U2)
    print("S:")
    print(S2)
    print("Vt:")
    print(Vt2)
    print("Decomposition matrix for M3:")
    print("U:")
    print(U3)
    print("S:")
    print(S3)
    print("Vt:")
    print(Vt3)
