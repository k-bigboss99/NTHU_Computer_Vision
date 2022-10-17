import numpy as np
from numpy import array
from matplotlib import pyplot as plt
from PIL import Image
import cv2

#from myHarrisCornerDetector import detect

"""
ref. https://www.safwan.xyz/2020/03/14/harris-corner.html
"""



def gaussian_mask(n, sigma=None):
    if sigma is None:
        sigma = 0.3 * (n // 2) + 0.8
    X = np.arange(-(n//2), n//2+1)
    kernel = np.exp(-(X**2)/(2*sigma**2))
    #gaussian_kernel_1 = (1/(2*Sigma_1**2*math.pi)) * (np.exp(-(x_1**2+y_1**2)/(2*Sigma_1**2)))
    return kernel

"""
sobel kernel 
[sobel_1,2*sobel_1,sobel_1] 
[sobel_2,0        ,sobel_2]
計算 Gx Gy 的梯度方向 
window size 來計算每個像素的梯度平方和
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
"""



def seperable_conv(I, filter_x, filter_y):
    h, w = I.shape[:2]
    n = filter_x.shape[0]//2
    I_a = np.zeros(I.shape)
    I_b = np.zeros(I.shape)
    for x in range(n, w-n):
        patch = I[:, x-n:x+n+1]
        I_a[:, x] = np.sum(patch * filter_x, 1)
    filter_y = np.expand_dims(filter_y, 1)

    for y in range(n, h-n):
        patch = I_a[y-n:y+n+1, :]
        I_b[y, :] = np.sum(patch * filter_y, 0)
    return I_b


def detect(I, n_g=10, n_w=3, k=0.06):
    h, w = I.shape
    sobel_1 = np.array([-1, 0, 1])
    sobel_2 = np.array([1, 2, 1])
    
    I_x = seperable_conv(I, sobel_1, sobel_2)
    I_y = seperable_conv(I, sobel_2, sobel_1)
    
    g_kernel = gaussian_mask(n_g)

    I_x = seperable_conv(I_x, g_kernel, g_kernel)
    I_y = seperable_conv(I_y, g_kernel, g_kernel)
    
    """
    泰勒式展開 
    M(D_temp)=[
        [IxIx,IxIy],
        [IyIx,IyIy]
    ]
    
    """

    D_temp = np.zeros((h, w, 2, 2))
    D_temp[:, :, 0, 0] = np.square(I_x) 
    D_temp[:, :, 0, 1] = I_x * I_y
    D_temp[:, :, 1, 0] = D_temp[:, :, 0, 1]
    D_temp[:, :, 1, 1] = np.square(I_y)

    g_filter = gaussian_mask(n_w)
    g_filter = np.dstack([g_filter] * 4).reshape(n_w, 2, 2) 
    D = seperable_conv(D_temp, g_filter, g_filter)
    P = D[:, :, 0, 0]  # IxIx
    Q = D[:, :, 0, 1]  # IxIy
    R = D[:, :, 1, 1]  # IyIy
    """
    A=[
        [a,b]
        [c,d]
    ]
    解 a+b/2 [+ or -](((a-d)平方+4bc))取平方根 再除2
    """
    T1 = (P + R) / 2   # a+b/2
    T2 = np.sqrt(np.square(P - R) + 4 * np.square(Q)) / 2 # squart 平方根 square 平方 , ((IxIx - IxIy) + 4(IxIy*IxIy) ) 開平方根 除2
    L_1 = T1 - T2  # ((IxIx+IyIy)/2)  - (((IxIx - IxIy) + 4(IxIy*IxIy)) 開平方根 除2 )
    L_2 = T1 + T2  # 
    C = L_1 * L_2 - k * np.square(L_1 + L_2)  # R=det(M)-k(trace(M))^{2}=\lambda _{1}\lambda _{2}-k(\lambda _{1}+\lambda _{2})^{2}}
    return C, I_x, I_y, L_1, L_2 


"""
main 
"""
# img_path =  r'C:\Users\user\Desktop\code\Harris-Corner-Python-master\imgs\chess.png'
# img_path = r'NTHU_CV_HW1/1a_notredame.jpg'

img_1 = cv2.imread("1a_notredame.jpg")
#img_1 = cv2.imread(r'NTHU_CV_HW1/chessboard-hw1.jpg')



# img_gray = array(Image.open(img_path).convert('L')) # convert gray imgage
img_gray = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
# img_gray = (img_gray - img_gray.min())/(img_gray.max()-img_gray.min())
# cv2.imwrite("img__test.jpg",img_gray)
C, I_x, I_y, L_1, L_2 = detect(img_gray, k=0.06) # k ∈ [ 0.04 , 0.06 ]
# C = (C - C.min())/(C.max()-C.min())



# plt.figure(figsize=(13, 5))
# plt.subplot(121)
# plt.title('$I_x$')
# plt.imshow(I_x, cmap='gray')
# plt.subplot(122)
# plt.title('$I_y$')
# plt.imshow(I_y, cmap='gray')
# plt.tight_layout()
# plt.show()






"""
show 
"""
plt.figure(figsize=(13, 5))
plt.subplot(121)
plt.title(r'$\lambda_1$')
plt.imshow(L_1, cmap='gnuplot')

#plt.savefig("L_1.jpg")

# plt.colorbar()
plt.subplot(122)
plt.title(r'$\lambda_2$')
plt.imshow(L_2, cmap='gnuplot')
# plt.colorbar()
plt.tight_layout()
plt.show()



plt.figure(figsize=(13, 5))
plt.subplot(121)
#plt.imshow(C-0.457, cmap='gnuplot')
plt.imshow(C, cmap='gnuplot')
plt.title('Corner-ness Map')
plt.subplot(122)
#plt.imshow(img_gray/2+2*C*(C >= 0.457), cmap='gnuplot')
plt.imshow((C >= 0.4), cmap='gnuplot')
plt.title('Detected Corners')
plt.tight_layout()
plt.show()