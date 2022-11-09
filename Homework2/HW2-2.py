import cv2
import numpy as np
import math
import copy

import matplotlib.pyplot as plt
from PIL import Image


# Get eight points coordinate : four points, four corresponding points
def get_eight_points_coordinate(event, x1, y1, flags, param):
    
    global mouse, count
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse += [[x1, y1]]
    elif event == cv2.EVENT_LBUTTONUP:
        if(count < 4):
            cv2.circle(image, (x1, y1), 8, (0, 0, 255), -1)
        elif(count >= 4):
            cv2.circle(image, (x1, y1), 8, (255, 0, 0), -1)
        
        cv2.imshow('image', image)
        
        count += 1
        if count >= num_points:
            cv2.destroyAllWindows()

# Compute the Homography_matrix : AH = 0
def compute_homography_matrix(coordinate):

    A = np.zeros((len(coordinate), 9))

    for i in range(4):

        x1 = coordinate[i][0]
        y1 = coordinate[i][1]

        x2 = coordinate[i+4][0]
        y2 = coordinate[i+4][1]

        A[i*2][0] = x1;       A[i*2][1] = y1;       A[i*2][2] = 1
        A[i*2][3] = 0;        A[i*2][4] = 0;        A[i*2][5] = 0
        A[i*2][6] = -x2 * x1; A[i*2][7] = -x2 * y1; A[i*2][8] = -x2

        A[i*2 + 1][0] = 0;        A[i*2 + 1][1] = 0;        A[i*2 + 1][2] = 0
        A[i*2 + 1][3] = x1;       A[i*2 + 1][4] = y1;       A[i*2 + 1][5] = 1
        A[i*2 + 1][6] = -y2 * x1; A[i*2 + 1][7] = -y2 * y1; A[i*2 + 1][8] = -y2

    eigenvalue = np.linalg.eig(np.dot(np.transpose(A), A))[0]
    eigenvector = np.linalg.eig(np.dot(np.transpose(A), A))[1]

    Homography_matrix = eigenvector[:, np.argmin(eigenvalue)]
    Homography_matrix = np.reshape(Homography_matrix, (3, 3))

    return Homography_matrix

# Using back ward warping and bilinear interpolation
def backward_and_bilinear(coordinate, Homography_matrix):

    def adjacent_area(polygon): 
        n = len(polygon)
        if n < 3:
            return 0
        vectors = np.zeros((n, 2))
        for i in range(0, n):
            vectors[i, :] = polygon[i, :] - polygon[0, :]
        area = 0
        for i in range(1, n):
            v1 = vectors[i - 1, :]
            v2 = vectors[i, :]
            area = area + (v1[0] * v2[1] - v1[1] * v2[0]) / 2
        
        return abs(area)

    def comput_polygon_area(points_cor, points):

        points_cor_polygon_area = adjacent_area(points_cor)
        
        area1 = adjacent_area(np.asarray([points_cor[0], points_cor[1], points]))
        area2 = adjacent_area(np.asarray([points_cor[1], points_cor[2], points]))
        area3 = adjacent_area(np.asarray([points_cor[2], points_cor[3], points]))
        area4 = adjacent_area(np.asarray([points_cor[3], points_cor[0], points]))
        total_area = area1 + area2 + area3 + area4

        return total_area, points_cor_polygon_area


    points = copy.copy(coordinate[:4,])
    points_cor = copy.copy(coordinate[4:,])

    points_min_x = points[:, 0][np.argmin(points[:, 0])]
    points_max_x = points[:, 0][np.argmax(points[:, 0])]

    points_min_y = points[:, 1][np.argmin(points[:, 1])]
    points_max_y = points[:, 1][np.argmax(points[:, 1])]

    points_cor_min_x = points_cor[:, 0][np.argmin(points_cor[:, 0])]
    points_cor_max_x = points_cor[:, 0][np.argmax(points_cor[:, 0])]

    points_cor_min_y = points_cor[:, 1][np.argmin(points_cor[:, 1])]
    points_cor_max_y = points_cor[:, 1][np.argmax(points_cor[:, 1])]

    image_backward = copy.copy(image)    

    for x1 in range(points_min_x, points_max_x):
        for y1 in range(points_min_y, points_max_y):

            area1, area2 = comput_polygon_area(points, [x1, y1])
            if (abs((area1 / area2 ) - 1) < 10**(-5)):
                x1y1 = np.asarray([x1, y1, 1])
                x2y2 = np.dot(Homography_matrix, np.transpose(x1y1))
                x2 = x2y2[0] / x2y2[2]
                y2 = x2y2[1] / x2y2[2]
                
                area3, area4 = comput_polygon_area(points_cor, [x2, y2])
                if (abs((area3 / area4 ) - 1) < 10**(-5)): 
                    
                    max_x = math.floor(x2)
                    max_y = math.floor(y2)
                    
                    array = []
                    array.extend([((max_x-x2)**2 + (max_y-y2)**2), ((max_x+1-x2)**2 + (max_y-y2)**2), 
                                ((max_x-x2)**2 + (max_y+1-y2)**2), ((max_x+1-x2)**2 + (max_y+1-y2)**2)])
                    
                    min_index = np.argmin(array)

                    if min_index == 0:
                        x2 = max_x; y2 = max_y
                    elif min_index == 1:
                        if max_x + 1 < weight:
                            x2 = max_x + 1; y2 = max_y
                    elif min_index == 2:
                        if max_y+ 1 < height:
                            x2 = max_x;  y2 = max_y + 1
                    elif min_index == 3:
                        if ((max_y+ 1) < height) & ((max_x + 1) < weight):
                            x2 = max_x + 1; y2 = max_y + 1
                        elif max_y+ 1 < height:
                            x2 = max_x; y2 = max_y + 1
                        elif max_x + 1 < weight:
                            x2 = max_x + 1; y2 = max_y

                    image_backward[y1][x1] = image[y2][x2]

    
    Homography_inv = np.linalg.inv(Homography_matrix)
               
    for x1 in range(points_cor_min_x, points_cor_max_x):
        for y1 in range(points_cor_min_y, points_cor_max_y):
            
            area_cor1, area_cor2= comput_polygon_area(points_cor, [x1, y1])
            
            if (abs((area_cor1 / area_cor2 ) - 1) < 10**(-5)):            
                x1y1 = np.asarray([x1, y1, 1])
                x2y2 = np.dot(Homography_inv, np.transpose(x1y1))
                x2 = x2y2[0] / x2y2[2]
                y2 = x2y2[1] / x2y2[2]
            
                area_cor3, area_cor4= comput_polygon_area(points, [x2, y2])
                
                if (abs((area_cor3 / area_cor4 ) - 1) < 10**(-5)):
                    
                    max_x = math.floor(x2)
                    max_y = math.floor(y2)
                    
                    array = []
                    array.extend([((max_x-x2)**2 + (max_y-y2)**2), ((max_x+1-x2)**2 + (max_y-y2)**2), 
                                ((max_x-x2)**2 + (max_y+1-y2)**2), ((max_x+1-x2)**2 + (max_y+1-y2)**2)])

                    min_index = np.argmin(array)

                    if min_index == 0:
                        x2 = max_x; y2 = max_y
                    elif min_index == 1:
                        if max_x + 1 < weight:
                            x2 = max_x + 1; y2 = max_y
                    elif min_index == 2:
                        if max_y + 1 < height:
                            x2 = max_x;  y2 = max_y + 1
                    elif min_index == 3:
                        if ((max_y+ 1) < height) & ((max_x + 1) < weight):
                            x2 = max_x + 1; y2 = max_y + 1
                        elif max_y+ 1 < height:
                            x2 = max_x; y2 = max_y + 1
                        elif max_x + 1 < weight:
                            x2 = max_x + 1; y2 = max_y

                    image_backward[y1][x1] = image[y2][x2]
    
    cv2.imwrite('rectification.jpg', image_backward)
                    

if __name__ == '__main__': 

    image = cv2.imread('Delta-Building.jpg')

    num_points = 8
    count = 0; mouse = []

    height = image.shape[0]
    weight = image.shape[1]

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', weight, height)
    cv2.setMouseCallback('image', get_eight_points_coordinate)

    cv2.imshow('image', image)
    cv2.waitKey(0)

    image_points = np.asarray(mouse)
    np.save('points', image_points)
    coordinate = np.load("points.npy")

    Homography_matrix = compute_homography_matrix(coordinate)
    print("Homography_matrix\n", Homography_matrix)
    
    backward_and_bilinear(coordinate, Homography_matrix)

