# README

-  file struture
    ![](https://i.imgur.com/hIRsbhJ.png)

- 使用的函式庫
```python=
import math
import cv2
import numpy as np 
import skimage.io
import skimage.color

from PIL import Image 
from scipy import signal
from email.mime import image
from unicodedata import name
from matplotlib import pyplot as plt
```

- `python HW1-1.py`
    - 以下參數簡稱`k=kernel`、`t=threshold`、`w=window_size`
    - Gaussian Smooth => def gaussian_smooth(image, kernel, sigma, name)
        - gaussian-smooth-A(k=10).jpg
        - gaussian-smooth-B(k=5).jpg
        - gaussian-smooth-B(k=10).jpg
    - Sobel edge detection => def sobel_edge_detection(image, threshold, name)
        - direction-A(k=5+t=30).jpg
        - direction-A(k=10+t=30).jpg
        - direction-B(k=5+t=30).jpg
        - direction-B(k=10+t=30).jpg
        - magnitude-A(k=5+t=30).jpg
        - magnitude-A(k=10+t=30).jpg
        - magnitude-B(k=5+t=30).jpg
        - magnitude-B(k=10+t=30).jpg
    - Structure Tensor => def structure_tensor(image, window_size, name)
        - structure-tensor-A(k=10+w=3).jpg
        - structure-tensor-A(k=10+w=5).jpg
        - structure-tensor-B(k=10+w=3).jpg
        - structure-tensor-B(k=10+w=5).jpg
    - Non-maximal Suppression => def nms(image, window_size, threshold, R, name)
        - NMS-A(w=3+t=100).jpg
        - NMS-A(w=5+t=100).jpg
        - NMS-B(w=3+t=100).jpg
        - NMS-B(w=5+t=100).jpg
    - Rotate and Scale => def rotate_and_scale(image, times, angle, name)
        - rotate_and_scaleA.jpg
        - rotate_and_scaleB.jpg

- `python HW1-2.py`
    - SIFT interest point detection and matching => def sift_detection_and_matching(image1, image2, point)
        - matching.jpg