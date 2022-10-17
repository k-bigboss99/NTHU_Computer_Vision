import math
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import cv2
from scipy import signal
'''
code explain follow the paper alg:
for function :gaussian_smooth()==>get_gaussian_filter==>get_gaussian_value
    1. smoothing the image
for function :structure_tensor_get_R_and_nms
    2. computing the gradient of image (gradient Ix,Iy ;Sobel ) 
    3. computing autocorrelation matrix (the Ix^2,Ix*Iy,Iy^2 & convoluted,structure tensor) 
    4. compute the corner strength(calculate the R) 
    5. Non-maximun supression
    6. Output
'''
def get_gaussian_value(sigma=1,x=1,y=1):
    '''
    Input:
        sigma : float,control scale of the gaussian function
        x : int,index
        y : int,index
    Output:
        gaussian value
    '''
    return (math.exp(-(x*x+y*y)/(2*sigma*sigma)))/(2*math.pi*sigma*sigma)
def get_gaussian_filter(sigma = 5, kernel_size = 5):
    '''
    Input:
        sigma : float,control scale of the gaussian function
        kernel_size : int,the filter size,more larger more smoother
    Output:
        gaussian_filter: the filter of gaussian
    '''
    mid = int(kernel_size/2) 
    gaussian_filter = np.zeros([kernel_size, kernel_size])
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_filter[i, j] = get_gaussian_value(sigma, i-mid, j-mid) 
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    return gaussian_filter
def gaussian_smooth(image_path,kernel_size=5,sigma=5):
    '''
    Input:
        image_path : path of image 
        sigma : float,control scale of the gaussian function
        kernel_size : int,the filter size,more larger more smoother
    Output:
        iamge blur with gaussian filter
    '''
    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)
    gaussian_filter = get_gaussian_filter(sigma, kernel_size)
    b5 = signal.convolve2d(b, gaussian_filter, boundary='symm', mode='same')
    g5 = signal.convolve2d(g, gaussian_filter, boundary='symm', mode='same')
    r5 = signal.convolve2d(r, gaussian_filter, boundary='symm', mode='same')
    output = cv2.merge([b5, g5, r5])
    cv2.imwrite('a_1_G_blur_k_size_'+str(kernel_size)+'.jpg', output) 
def structure_tensor_get_R(window_size,image_path, threshold):
    Hx = np.zeros([3, 3])
    Hx[0] = [-1,0,1]
    Hx[1] = [-2,0,2]
    Hx[2] = [-1,0,1]  
    Hy = np.zeros([3, 3])
    Hy[0] = [1,2,1] 
    Hy[1] = [0,0,0]
    Hy[2] = [-1,-2,-1]
    image = cv2.imread(image_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_shape = np.shape(gray)
    fx = signal.convolve2d(gray, Hx / 8, mode='same')
    fy = signal.convolve2d(gray, Hy / 8, mode='same')
    # 3. computing autocorrelation matrix (structure tensor)
    gaussian_filter = get_gaussian_filter(sigma=5, kernel_size=window_size)  
    fx_fx = fx * fx
    fx_fy = fx * fy
    fy_fy = fy * fy
    fx_fx = signal.convolve2d(fx_fx, gaussian_filter, mode='same')
    fx_fy = signal.convolve2d(fx_fy, gaussian_filter, mode='same')
    fy_fy = signal.convolve2d(fy_fy, gaussian_filter, mode='same')
    #  4. compute the corner strength(calculate the R)
    R = (fx_fx * fy_fy - fx_fy * fx_fy) / (fx_fx + fy_fy)
    skip = np.zeros(np.shape(fx_fx))
    print("123")
    print(skip.shape)
    for i in range(gray_shape[0]):
        for j in range(gray_shape[1]):
            if R[i, j] < threshold:
                skip[i, j] = 1
    return skip,R,gray_shape
# 5. Non-maximun supression follow the alg 
def nms(tensor_window_size,nms_window_size, threshold, image_path, skip, R, gray_shape):
    image = cv2.imread(image_path) 
    # 5. Non-maximun supression follow the alg
    radius = 0 
    if nms_window_size % 2==1:
        radius = int((nms_window_size - 1) / 2)
    else:
        radius = int(nms_window_size / 2)
    for i in range(radius - 1, gray_shape[0] - radius):
        j = radius - 1
        while ((j < (gray_shape[1] - radius)) and (skip[i, j] or (R[i, j - 1] > R[i, j]))):
            j = j + 1
        while (j < (gray_shape[1] - radius)):
            while ((j < (gray_shape[1] - radius)) and (skip[i, j] or (R[i, j + 1] > R[i, j]))):
                j = j + 1
            if (j < (gray_shape[1] - radius)):
                p1 = j + 2
                while (p1 <= (j + radius) and (R[i, p1] < R[i, j])):
                    skip[i, p1] = 1
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
                                    skip[k, l] = 1
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
                            cv2.circle(image, (j, i), 1, (0, 0, 255), -1)
                j = p1
    cv2.imwrite('a_3_window_size_' + tensor_window_size + 'threshold_' + str(threshold) + '.jpg',
                image)

if __name__ == '__main__':
    image_path = "1a_notredame.jpg"
    image = cv2.imread("1a_notredame.jpg")
    print(image.shape)
    threshold = 4
    gaussian_smooth(image_path, kernel_size=10, sigma=5)
    skip, R, gray_shape = structure_tensor_get_R(window_size=3, image_path="a_1_G_blur_k_size_10.jpg", threshold=threshold)
    nms(tensor_window_size="3", nms_window_size=3, threshold=threshold, image_path="a_1_G_blur_k_size_10.jpg", skip=skip, R=R,
        gray_shape=gray_shape)
    skip, R, gray_shape = structure_tensor_get_R(window_size=30, image_path="a_1_G_blur_k_size_10.jpg", threshold=threshold)
    nms(tensor_window_size="30", nms_window_size=3, threshold=threshold, image_path="a_1_G_blur_k_size_10.jpg", skip=skip, R=R,
        gray_shape=gray_shape)