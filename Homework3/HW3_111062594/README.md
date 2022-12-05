# README

-  file struture
    ![](https://i.imgur.com/U3CC5Vx.png =500x)


- 使用的函式庫
```python=
import cv2
import numpy as np
import random
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from matplotlib import colors
```

- `python HW3-1.py`
    - SIFT(Establish point correspondences between the SIFT feature points detected from the single-book images and the cluttered-book image)
        - def sift_detection_and_matching(image1, image2, point, threshold)
            - sift-1-book1.jpg
            - sift-1-book2.jpg
            - sift-1-book3.jpg
    - RANSAC(Apply the RANSAC program to find the best matching homography transformation between each single-book image and the input image)
        - def ransac_homography_transformation(image1, image2, obj_bbox, num_iterations, threshold)
            - ransac-1-book1.jpg
            - ransac-1-book2.jpg
            - ransac-1-book3.jpg
            - deviation.jpg

- `python HW3-2.py`
    - K-means(Apply K-means on the image (RGB color space) and try it with three different K values (your K should be > 3) and show the results)
        - def k_means_segmentation(img, pixels, K_clusters, times)
            - K-means with 5 clusters on 2-image.jpg
            - K-means with 7 clusters on 2-image.jpg
            - K-means with 9 clusters on 2-image.jpg
            - K-means with 5 clusters on 2-masterpiece.jpg
            - K-means with 7 clusters on 2-masterpiece.jpg
            - K-means with 9 clusters on 2-masterpiece.jpg
    - Implement K-means++ to have better initial guess 
        - def k_means_plus_segmentation(img, pixels, K_clusters, times)
            - K-means-plus with 5 clusters on 2-image.jpg
            - K-means-plus with 7 clusters on 2-image.jpg
            - K-means-plus with 9 clusters on 2-image.jpg
            - K-means-plus with 5 clusters on 2-masterpiece.jpg
            - K-means-plus with 7 clusters on 2-masterpiece.jpg
            - K-means-plus with 9 clusters on 2-masterpiece.jpg
    - Implement the mean-shift algorithm to segment the same colors in the target image. 
        - def uniform_kernel_rgbspace(img, image)
            - uniform-kernel 2-image.jpg
            - uniform-kernel Meanshift with bandwidth 0.3 on 2-image.jpg
            - uniform-kernel Meanshift with bandwidth 0.4 on 2-image.jpg
            - uniform-kernel Meanshift with bandwidth 0.5 on 2-image.jpg
            - uniform-kernel 2-masterpiece.jpg
            - uniform-kernel Meanshift with bandwidth 0.3 on 2-masterpiece.jpg
            - uniform-kernel Meanshift with bandwidth 0.4 on 2-masterpiece.jpg
            - uniform-kernel Meanshift with bandwidth 0.5 on 2-masterpiece.jpg
    - Show the mean-shift segmentation results with three different sets of bandwidth parameters.
        - def meanshift_segmentation(img, pixels, bandwidth)
            - Meanshift with bandwidth 0.3 on 2-image.jpg
            - Meanshift with bandwidth 0.4 on 2-image.jpg
            - Meanshift with bandwidth 0.5 on 2-image.jpg
            - Meanshift with bandwidth 0.3 on 2-masterpiece.jpg
            - Meanshift with bandwidth 0.4 on 2-masterpiece.jpg
            - Meanshift with bandwidth 0.5 on 2-masterpiece.jpg