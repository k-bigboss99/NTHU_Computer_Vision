import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from matplotlib import colors


def kmeans_segmentation(img, pixels, K_clusters, times):

    #initialization
    height = img.shape[0]
    weight = img.shape[1]

    # pixel_clusters (each pixel belongs to cluster i)
    pixel_clusters = np.zeros((height * weight), dtype=int)
    M  = pixels.shape[0]

    # randomly choose K center_clusters 
    idx_random = np.random.choice(M, K_clusters, replace=False) 
    center_clusters = pixels[idx_random]
    
    # update K center_clusters value
    new_center_clusters = np.zeros_like(center_clusters)

    count = 0
    while(times):
        for i in range(pixels.shape[0]):
            # pixel_clusters (each pixel belongs to cluster i) : shortest distances
            idx = np.argsort(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            pixel_clusters[i] = idx[0]

        for j in range(K_clusters):
            # update K center_clusters value : comput all pixel in cluster i, and get means.
            clusters_class = np.where(pixel_clusters == j)[0]
            new_center_clusters[j] = np.sum(pixels[clusters_class], axis=0) / clusters_class.shape[0]

        return pixel_clusters.reshape(height, weight)


def kmeans_plusplus_segmentation(img, pixels, K_clusters, times):
    #initialization
    height = img.shape[0]
    weight = img.shape[1]

    # pixel_clusters (each pixel belongs to cluster i)
    pixel_clusters = np.zeros((height * weight), dtype=int)
    M  = pixels.shape[0]
    
    # randomly choose one center_clusters 
    idx_random = np.random.choice(M, 1, replace=False) 
    center_clusters = pixels[idx_random]

    # choose the next center_clusters 
    i = 0; j = 0
    for j in range(K_clusters-1):
        distanceList = []
        for i in range(pixels.shape[0]):
            # pixel_clusters (each pixel belongs to cluster i) : shortest distances
            idx = np.argsort(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            distance = sum(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            
            distanceList.append(distance)

        max_distance = max(distanceList)
        idx_random = np.append(idx_random, distanceList.index(max_distance))
        center_clusters = pixels[idx_random]


    # update K center_clusters value
    new_center_clusters = np.zeros_like(center_clusters)


    while(times):
        i = 0; j = 0
        for i in range(pixels.shape[0]):
            # pixel_clusters (each pixel belongs to cluster i) : shortest distances
            idx = np.argsort(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            pixel_clusters[i] = idx[0]

        for j in range(K_clusters):
            # update K center_clusters value : comput all pixel in cluster i, and get means.
            clusters_class = np.where(pixel_clusters == j)[0]
            new_center_clusters[j] = np.sum(pixels[clusters_class], axis=0) / clusters_class.shape[0]

        return pixel_clusters.reshape(height, weight)


def draw_clusters_on_image(img, pixel_clusters):
    
    K_clusters = int(pixel_clusters.max()) + 1
    
    average_color = np.zeros((K_clusters, 3))
    
    cluster_count = np.zeros(K_clusters)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = pixel_clusters[i,j]
            cluster_count[c] += 1

            average_color[c, :] += img[i, j, :]

    for c in range(K_clusters):
        average_color[c,:] /= float(cluster_count[c])

    out_im = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = pixel_clusters[i,j]
            out_im[i,j,:] = average_color[c,:]

    return out_im

def meanshift_segmentation(img, pixels, bandwidth):
    
    #initialization
    height,weight,_ = img.shape
    record = np.ones([height*weight],dtype=int)

    cluster_means = np.empty((0,pixels.shape[1]))

    pixel_clusters = np.zeros(pixels.shape[0],dtype=int)
    
    idx_rand_no_seen_record = []
    idx_i = 0
    timer=0

    while np.sum(record) > 0:
        
        # randomly choose one feature from haven't seen
        idx = np.where(record > 0)[0]
        # print(idx)
        idx_rand_no_seen = idx[np.random.choice(idx.shape[0], 1)]

        # print(idx_rand_no_seen) # 找一些點大概不到 50點 就可以收斂

        mean = pixels[idx_rand_no_seen].flatten()# flatten() 少一維度 
        # print(mean) 
        flag = True
        
        # mean shift
        while flag:
            dis = np.linalg.norm(pixels - mean,axis=1)
            idx_within = np.where(dis < bandwidth)[0] # 返回 index

            # import pdb;pdb.set_trace()
            new_mean = np.sum(pixels[idx_within],axis=0) / idx_within.shape[0]
        #     If the output mean vector from the mean shift step is
        #       sufficiently close (within half a bandwidth) to another cluster
        #       center, say it's part of that cluster
            if np.linalg.norm(new_mean-mean) < bandwidth:  #norm 相減 平方 sum 再開根號
                flag = False
                # idx_i+=1
                # print(timer)
                # timer+=1
            else:
                # If it's not sufficiently close to any other cluster center, make a new cluster
                mean = new_mean.copy()
                # print("bad")
            record[idx_within] = 0
            
        mean_dis = np.linalg.norm(cluster_means - new_mean,axis=1)

        # 每一個 cluster_means 隨著找到的 idx_rand_no_seen 的點 當成 cluster 的 中心算 mean 
        # print(mean_dis)

        """
        把 height 範圍內 點都統一 ,且算完的值 給 mean_dis
        取 最小 mean_dis[0] ,找真正有用的cluster
        """
        if mean_dis.size > 0 and mean_dis[np.argsort(mean_dis)[0]] < bandwidth / 2:
            # pixel_clusters[idx_within] = mean_dis[np.argsort(mean_dis)[0]] # 是給 index
            pixel_clusters[idx_within] = np.argsort(mean_dis)[0]
            # print(pixel_clusters[idx_within])
            # print(np.argsort(mean_dis)[0])
            # print(timer)# 3次
            # timer+=1
        else:
            cluster_means = np.vstack((cluster_means,new_mean)) # 矩陣矩陣堆疊 直方向 
            # print(timer)# 29次
            # timer+=1

            # idx_within 的值 = 總共數量的值 ,全部該次的idx_within 都同個id 然後給 plot 再算idx_within 全部的平均
            pixel_clusters[idx_within] = cluster_means.shape[0] 
    # print(cluster_means.shape[0])
    # print(idx_rand_no_seen_record)
    
    return pixel_clusters.reshape(height,weight)
   
def show_rgbspace(img):
    r, g, b = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

if __name__ == '__main__':

    # K_clusters = [5,10,15]
    K_clusters = [5]
    # bandwidths = [0.3,0.5,0.6]
    # bandwidths = [0.3]

    image = '2-image.jpg'
    img = imageio.imread(image)

    pixels = np.zeros((img.shape[0] * img.shape[1], 5))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixels[i*img.shape[1] + j, :] = np.array([i, j,img[i, j, 0], img[i, j, 1], img[i, j, 2]])

    pixels_normalized = pixels / pixels.max(axis = 0)

    # Segmentation using K-Means
    for nc in K_clusters:
        
        K_clustered_pixels = kmeans_segmentation(img, pixels_normalized, nc, times=50)
        Kplus_clustered_pixels = kmeans_plusplus_segmentation(img, pixels_normalized, nc, times=50)
        
        K_cluster_img = draw_clusters_on_image(img, K_clustered_pixels)
        Kplus_cluster_img = draw_clusters_on_image(img, Kplus_clustered_pixels)
        
        imageio.imsave('K-means with %d clusters on %s' % (int(nc), image), K_cluster_img)
        imageio.imsave('K-means-plus with %d clusters on %s' % (int(nc), image),  Kplus_cluster_img)


        # # Part II: Segmentation using Meanshift
        # for bandwidth in bandwidths:

        #     clustered_pixels = meanshift_segmentation(img, pixels_normalized, bandwidth)
        #     cluster_img = draw_clusters_on_image(img, clustered_pixels)

        #     # for i in range(3):
        #     imageio.imsave('output/Meanshift_with_bandwidth_%.2f_on_%s.jpg' % (bandwidth,filename),cluster_img)
        #     plt.imshow(cluster_img)
        #     plt.title('Meanshift with bandwidth %.2f on %s.jpg' % (bandwidth, filename))
        #     plt.show()

    # show original

    # for filename in ['2-image', '2-masterpiece']:#, '2-masterpiece'
    #     print("original: %s.jpg" % filename)
    #     img_1 = imageio.imread('%s.jpg' % filename)
    #     show_rgbspace(img_1)
    #     for bandwidth in bandwidths:
    #         img = imageio.imread('output/Meanshift_with_bandwidth_%.2f_on_%s.jpg' % (bandwidth,filename))
            

            
    #         print('Meanshift with bandwidth %.2f on %s.jpg' % (bandwidth, filename))
    #         show_rgbspace(img)