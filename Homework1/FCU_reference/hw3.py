from cv2 import cv2
import numpy as np
import math

# 分別讀取 picture1-3(兩張建築、一張自拍)
img = cv2.imread('picture3.jpg',0)
# 讀取簽名檔
signature = cv2.imread('sign.png',0)
# 模糊化濾掉高頻/雜訊/細節
blur_img = cv2.GaussianBlur(img,(3,3),0)

# 使用Sobel計算梯度(包含量值和方向性)
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
# 梯度量值計算
magnitude = np.abs(gx) + np.abs(gy)
# 梯度方向計算
theta = np.degrees(np.arctan2(gy,gx))

# 非最大值抑制 (non maximal suppression, NMS)
height = magnitude.shape[0]
weight = magnitude.shape[1]
edge = np.zeros_like(magnitude)

for i in range(1,height - 1):
    for j in range(1,weight - 1):
        # 四個方向(上下、左右、正45度、負45度)，只保留同方向上連續點中的最大值

        if ( ( (theta[i,j] >= -22.5) and (theta[i,j]< 22.5) ) or
                ( (theta[i,j] <= -157.5) and (theta[i,j] >= -180) ) or
                ( (theta[i,j] >= 157.5) and (theta[i,j] < 180) ) ):
            magnitude_max = max(magnitude[i, j - 1], magnitude[i, j], magnitude[i, j + 1])
            edge[i, j] = magnitude[i, j]

        elif ( ( (theta[i,j] >= 22.5) and (theta[i,j]< 67.5) ) or
                ( (theta[i,j] <= -112.5) and (theta[i,j] >= -157.5) ) ):
            magnitude_max = max(magnitude[i - 1, j - 1], magnitude[i, j], magnitude[i + 1, j + 1])
            edge[i, j] = magnitude[i, j]

        elif ( ( (theta[i,j] >= 67.5) and (theta[i,j]< 112.5) ) or
                ( (theta[i,j] <= -67.5) and (theta[i,j] >= -112.5) ) ):
            magnitude_max = max(magnitude[i - 1, j], magnitude[i, j], magnitude[i + 1, j])
            edge[i, j] = magnitude[i, j]

        elif ( ( (theta[i,j] >= 112.5) and (theta[i,j]< 157.5) ) or
                ( (theta[i,j] <= -22.5) and (theta[i,j] >= -67.5) ) ):
            magnitude_max = max(magnitude[i + 1, j - 1], magnitude[i, j], magnitude[i - 1, j + 1])
            edge[i, j] = magnitude[i, j]

# 雙門檻與連通成份分析，高低門檻為200、100(2:1)
height = edge.shape[0]
weight = edge.shape[1]
canvas = np.zeros_like(edge)
for i in range(height):
    for j in range(weight):
        if edge[i,j] >= 200:
            canvas[i,j] = 255
        elif edge[i,j] <= 100:
            canvas[i,j] = 0
        elif (( edge[i+1,j] < 200) or (edge[i-1,j] < 200 )or( edge[i,j+1] < 200 )or
            (edge[i,j-1] < 200) or (edge[i-1, j-1] < 200 )or ( edge[i-1, j+1] < 200) or
                ( edge[i+1, j+1] < 200 ) or ( edge[i+1, j-1] < 200) ):
            canvas[i,j] = 255
canvas = np.uint8(edge)

# 去背簽名檔(size = 200*125)
height = canvas.shape[0]
weight = canvas.shape[1]
canvas[height - 125:,weight - 200:] = cv2.bitwise_or(canvas[height - 125:,weight - 200:],signature)

# Hough Transform
img2 = canvas.copy( )
# 參數1 : 灰度圖、參數2與 : 分別是\rho和\theta的精確度、參數4:閾值T
lines = cv2.HoughLines( img2, 1, math.pi/180.0, 135 )
if lines is not None:
	a,b,c = lines.shape
	for i in range( a ):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		a = math.cos( theta )
		b = math.sin( theta )
		x0, y0 = a*rho, b*rho
        # 由引數空間向實際座標點轉換
		pt1 = ( int(x0 + 1000*(-b)), int(y0 + 1000*(a)) )
		pt2 = ( int(x0 - 1000*(-b)), int(y0 - 1000*(a)) )
		cv2.line( img2, pt1, pt2, ( 255, 0, 0 ), 1)

# cv2.imwrite('canny3.jpg',canvas)
# cv2.imwrite('Hough3.jpg',img2)

cv2.imshow('canny',canvas)
cv2.imshow( "Hough", img2 )

cv2.waitKey(0)
cv2.destroyAllWindows()