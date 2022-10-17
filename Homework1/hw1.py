from email.mime import image
import math
import numpy as np
import cv2
from scipy import signal
import skimage.color  
import skimage.io 
from matplotlib import pyplot as plt
from PIL import Image 

np.seterr(divide='ignore',invalid='ignore')

# Gaussian Smooth
def gaussian_smooth(image, kernel, sigma, name):

    img = cv2.imread(image)
    b, g, r = cv2.split(img)    
    
    gaussian_filter = np.zeros([kernel, kernel])
    for i in range(kernel):
        for j in range(kernel):
            u = i - kernel / 2
            v =  j - kernel / 2
            gaussian_filter[i, j] = math.exp(-(u*u+v*v)/(2*sigma*sigma)) / (2*math.pi*sigma*sigma)
    gaussian_filter = gaussian_filter / gaussian_filter.sum()    
    
    B = signal.convolve2d(b, gaussian_filter, boundary='symm', mode='same')
    G = signal.convolve2d(g, gaussian_filter, boundary='symm', mode='same')
    R = signal.convolve2d(r, gaussian_filter, boundary='symm', mode='same')
    GS_image = cv2.merge([B, G, R])
    cv2.imwrite('GS_'+str(name)+'_k'+str(kernel)+'.jpg', GS_image)

# Sobel edge detection
def sobel_edge_detection(image, threshold, name):
    
    kernel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) 
    kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    
    image = cv2.imread(image) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = signal.convolve2d(gray, kernel_x, boundary='symm', mode='same')
    gy = signal.convolve2d(gray, kernel_y, boundary='symm', mode='same') 
    
    ## Calculate Magnitude
    magnitude =  np.abs(gx) + np.abs(gy)
    magnitude[magnitude < threshold] = 0
    cv2.imwrite('M'+ '_' + str(name) + '_t'  + str(threshold) + '.jpg', magnitude)
 
    ## Calculate Direction
    height = np.shape(image)[0]
    weight = np.shape(image)[1]
    hsv = np.zeros((height,weight,3))
    hsv[..., 0 ] = (np.arctan2(gy,gx)+np.pi)/(2*np.pi)
    hsv[..., 1 ] = np.ones((height, weight))
    hsv[..., 2 ] = magnitude / magnitude.max()- magnitude.min()

    HSV = skimage.color.hsv2rgb(hsv) 
    skimage.io.imsave('D'+ '_' + str(name) + '.jpg', HSV)  

def structure_tensor(image, window_size, name):
    kernel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) 
    kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) 

    image = cv2.imread(image) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = signal.convolve2d(gray, kernel_x / 8, mode='same')
    gy = signal.convolve2d(gray, kernel_y / 8, mode='same')

    sigma = 5
    gaussian_filter = np.zeros([window_size, window_size])
    for i in range(window_size):
        for j in range(window_size):
            u = i - window_size / 2
            v =  j - window_size / 2
            gaussian_filter[i, j] = math.exp(-(u*u+v*v)/(2*sigma*sigma)) / (2*math.pi*sigma*sigma)
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    
    Ixx = gx * gx
    Ixy = gx * gy
    Iyy = gy * gy
    Ixx = signal.convolve2d(Ixx, gaussian_filter, mode='same')
    Ixy = signal.convolve2d(Ixy, gaussian_filter, mode='same')
    Iyy = signal.convolve2d(Iyy, gaussian_filter, mode='same')

    R = (Ixx * Iyy - Ixy * Ixy) / (Ixx + Iyy)

    lambda_1 = (Ixx+Iyy)/2-np.sqrt(np.square(Ixx-Iyy)+4*np.square(Ixy))/2
    lambda_2 = (Ixx+Iyy)/2+np.sqrt(np.square(Ixx-Iyy)+4*np.square(Ixy))/2
    
     
    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    plt.title(r'$\lambda_1$'+name+'_window:'+str(window_size))
    plt.imshow(lambda_1, cmap='gnuplot')

    plt.subplot(122)
    plt.title(r'$\lambda_2$'+'_window:'+str(window_size))
    plt.imshow(lambda_2, cmap='gnuplot')

    plt.tight_layout()
    plt.show()
    return R

# 5. Non-maximun supression follow the alg 
def nms(image, window_size, threshold, R, name):
    image = cv2.imread(image) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if window_size % 2==1:
        radius = int((window_size - 1) / 2)
    else:
        radius = int(window_size / 2)

    skip = np.zeros(np.shape(gray))
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if R[i, j] < threshold:
                skip[i, j] = True
            else:
                skip[i, j] = False
    
    gray_shape = np.shape(gray)
    for i in range(radius, gray_shape[0] - radius):
        j = radius - 1
        
        while ((j < (gray_shape[1] - radius)) and (skip[i, j] or (R[i, j - 1] > R[i, j]))):
            j = j + 1
        
        while (j < (gray_shape[1] - radius)):
            while ((j < (gray_shape[1] - radius)) and (skip[i, j] or (R[i, j + 1] > R[i, j]))):
                j = j + 1
            if (j < (gray_shape[1] - radius)):
                p1 = j + 2
                while (p1 <= (j + radius) and (R[i, p1] < R[i, j])):
                    skip[i, p1] = True
                    p1 = p1 + 1
                if p1 > (j + radius):
                    p2 = j - 1
                    while ((p2 >= (j - radius)) and (R[i, p2] <= R[i, j])):
                        p2 = p2 - 1
                    if p2 < j - radius:
                        k = i + radius
                        found = False
                        while not found and k > i:
                            l = j + radius
                            while not found and (l >= (j - radius)):
                                if R[k, l] > R[i, j]:
                                    found = True
                                else:
                                    skip[k, l] = True
                                l = l - 1
                            k = k - 1
                        k = i - radius
                        while not found and k < i:
                            l = j - radius
                            while not found and (l <= (j + radius)):
                                if R[k, l] >= R[i, j]:
                                    found = True
                                l = l + 1
                            k = k + 1
                        if not found: 
                            cv2.circle(image, (j, i), 1, (0, 0, 255), 3)
                j = p1

    cv2.imwrite('NMS_' + name + '_' +str(window_size) + '+t' + str(threshold) + '.jpg', image)



if __name__ == '__main__': 
    
    image1 = "1a_notredame.jpg"
    image2 = "chessboard-hw1.jpg"

    # gaussian_smooth(image1, kernel=5, sigma=5, name='A')
    # gaussian_smooth(image1, kernel=10, sigma=5, name='A')
    # gaussian_smooth(image2, kernel=5, sigma=5, name='B')
    # gaussian_smooth(image2, kernel=10, sigma=5, name='B')
    
    # sobel_edge_detection(image1, threshold=30, name='A')
    # sobel_edge_detection(image2, threshold=30, name='B')

    R1 = structure_tensor(image1, window_size=3, name='A')
    R2 = structure_tensor(image1, window_size=5, name='A')
    R3 = structure_tensor(image2, window_size=3, name='B')
    R4 = structure_tensor(image2, window_size=5, name='B')

    nms(image1, window_size=3, threshold=100, R=R1, name='A')
    nms(image1, window_size=5, threshold=100, R=R2, name='A')
    nms(image2, window_size=3, threshold=100, R=R3, name='B')
    nms(image2, window_size=5, threshold=100, R=R4, name='B')