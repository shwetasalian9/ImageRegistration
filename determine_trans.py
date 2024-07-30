# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:52:41 2023

@author: scorp
"""

import numpy as np

def determine_transformation(matrix):
    # Check for translation
    if np.count_nonzero(matrix[:-1, :-1]) == 0 and np.count_nonzero(matrix[:-1, -1]) > 0:
        return "Translation"

    # Check for scaling
    if np.count_nonzero(matrix[:-1, :-1]) == matrix[:-1, :-1].size and np.count_nonzero(matrix[:-1, -1]) == 0:
        return "Scaling"

    # Check for rotation around x-axis
    if np.array_equal(matrix, np.array([[1, 0, 0, 0],
                                        [0, np.cos(theta), -np.sin(theta), 0],
                                        [0, np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 0, 1]])):
        return "Rotation around x-axis"

    # Check for rotation around y-axis
    if np.array_equal(matrix, np.array([[np.cos(theta), 0, np.sin(theta), 0],
                                        [0, 1, 0, 0],
                                        [-np.sin(theta), 0, np.cos(theta), 0],
                                        [0, 0, 0, 1]])):
        return "Rotation around y-axis"

    # Check for rotation around z-axis
    if np.array_equal(matrix, np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                        [np.sin(theta), np.cos(theta), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])):
        return "Rotation around z-axis"

    # If none of the above, it might be a combination of transformations or other type
    return "Other"

# Example usage
matrix = np.array([[1, 0, 0, 3],
                   [0, 1, 0, 4],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

transformation_type = determine_transformation(matrix)
print("Transformation type:", transformation_type)
