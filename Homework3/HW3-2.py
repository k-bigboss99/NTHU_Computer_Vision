import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
# from scipy.misc import imread
import imageio
from scipy.spatial.distance import cdist

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2

def kmeans_segmentation(im, features, num_clusters):

    #initialization
    times=0
    h,w,_ = im.shape
    # pixel 屬於 i 類
    pixel_clusters = np.zeros((h*w),dtype=int)
    M  = features.shape[0]

    # randomly choose
    idx_random = np.random.choice(M, num_clusters, replace=False) 
    centroids = features[idx_random]
    

    new_centriods = np.zeros_like(centroids)
    while True:
        # 
        for i in range(features.shape[0]):
            idx = np.argsort(np.linalg.norm(features[i] - centroids,axis=1)) #new_centriods
            pixel_clusters[i] = idx[0]

        for j in range(num_clusters):
           
            candidate = np.where(pixel_clusters == j)[0]
            # print(candidate)
            # print(j)
            new_centriods[j] = np.sum(features[candidate],axis=0)/candidate.shape[0] #算means 也可以用means
            # print(new_centriods[j])
            # print(j)
        
        # print(candidate)
        # print(candidate.shape)

        
        if np.allclose(new_centriods,centroids) and times >=50: # allclose 至少做超過50次 當中心點都不動時返回 pixel_clusters
            print("做了 %d 的次數" %(times))
            return pixel_clusters.reshape(h,w)

        else:
            centroids = new_centriods.copy()
            times+=1

def kmeans_plusplus_segmentation(im, features, num_clusters):

    #initialization
    times=0
    h,w,_ = im.shape
    pixel_clusters = np.zeros((h*w),dtype=int)
    M  = features.shape[0]
    
    tmp_nc = 2 
    dis_num=[] 
    num_tmp=0 # 計算過後的值
    dis_tmp=0 #  儲存暫時
    times=0

    # randomly choose
    idx_random = np.random.choice(M, num_clusters, replace=False) 

    centroids = features[idx_random[:2]]
    # print(centroids)

    new_centriods = np.zeros_like(centroids)

    # 找群中心
    while tmp_nc <= num_clusters-1:
        dis_tmp=0
        # num_tmp=0
        dis_num=0
        times=0
        for k in range(features.shape[0]):#features.shape[0]
            dis_num = np.sqrt((features[k] - centroids)**2)
            # print(dis_num)
            for j in range(tmp_nc):
                for i in range(5):
                    num_tmp += dis_num[j][i]
            
            if num_tmp > dis_tmp:
                dis_tmp = num_tmp
                times = k
                # print(num_tmp)
            num_tmp=0
        print(times) # 找到第二個之後第三個點最遠在哪
        # print(features[times])
        # print(centroids)
        centroids = np.append(centroids,features[times])
        centroids = centroids.reshape((tmp_nc+1,5))
        tmp_nc+=1
    
    # print(centroids)

    # while True:
    for i in range(features.shape[0]):
        idx = np.argsort(np.linalg.norm(features[i] - centroids,axis=1)) #new_centriods
        pixel_clusters[i] = idx[0]
        # for j in range(tmp_nc):
        #     candidate = np.where(pixel_clusters == j)[0]
        #     new_centriods[j] = np.sum(features[candidate],axis=0)/candidate.shape[0] #算means 也可以用means
    # print(new_centriods)
    return pixel_clusters.reshape(h,w)
        
        # if np.allclose(new_centriods,centroids) and times >=50: # allclose 至少做超過50次 當中心點都不動時返回 pixel_clusters
        #     print("做了 %d 的次數" %(times))
        #     return pixel_clusters.reshape(h,w)

        # else:
        #     centroids = new_centriods.copy()
        #     times+=1
    
 
def draw_clusters_on_image(im, pixel_clusters):
    num_clusters = int(pixel_clusters.max()) + 1
    
    average_color = np.zeros((num_clusters, 3))
    
    cluster_count = np.zeros(num_clusters)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            c = pixel_clusters[i,j]
            cluster_count[c] += 1

            average_color[c, :] += im[i, j, :]

    for c in range(num_clusters):
        average_color[c,:] /= float(cluster_count[c])

    out_im = np.zeros_like(im)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            c = pixel_clusters[i,j]
            out_im[i,j,:] = average_color[c,:]

    return out_im

if __name__ == '__main__':

    # Change these parameters to see the effects of K-means and Meanshift
    # num_clusters = [5, 10, 15, 20]
    num_clusters = [5,10,15]
    # bandwidths = [0.3]

    for filename in ['2-image', '2-masterpiece']:
        img = imageio.imread('%s.jpg' % filename)
        # img = imread('data/%s.jpeg' % filename)

        # Create the feature vector for the images(x.y.r.b.g)
        features = np.zeros((img.shape[0] * img.shape[1], 5))
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                features[row*img.shape[1] + col, :] = np.array([row, col,img[row, col, 0], img[row, col, 1], img[row, col, 2]])
        features_normalized = features / features.max(axis = 0)

        # Part I: Segmentation using K-Means
        for nc in num_clusters:
            clustered_pixels = kmeans_segmentation(img, features_normalized, nc)
            # clustered_pixels = kmeans_plusplus_segmentation(img, features_normalized, nc)
            cluster_im = draw_clusters_on_image(img, clustered_pixels)
            

            imageio.imsave('K-means_with_%d_clusters_on_%s.jpg' % (int(nc), filename),cluster_im)
            # imageio.imsave('2/output/clusters_on.jpg',cluster_im)
            plt.imshow(cluster_im)
            plt.title('K-means with %d clusters on %s.jpg' % (int(nc), filename))
            plt.show()