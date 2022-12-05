import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from matplotlib import colors


def k_means_segmentation(img, pixels, K_clusters, times):

    # initialization
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
    while (times):

        for i in range(pixels.shape[0]):
            # pixel_clusters (each pixel belongs to cluster i) : shortest distances
            idx = np.argsort(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            pixel_clusters[i] = idx[0]

        for j in range(K_clusters):
            # update K center_clusters value : comput all pixel in cluster i, and get means.
            clusters_class = np.where(pixel_clusters == j)[0]
            new_center_clusters[j] = np.sum(pixels[clusters_class], axis=0) / clusters_class.shape[0]

        times -= 1
        return pixel_clusters.reshape(height, weight)

def k_means_plus_segmentation(img, pixels, K_clusters, times):
    # initialization
    height = img.shape[0]
    weight = img.shape[1]

    # pixel_clusters (each pixel belongs to cluster i)
    pixel_clusters = np.zeros((height * weight), dtype=int)
    M  = pixels.shape[0]
    
    # randomly choose one center_clusters 
    idx_random = np.random.choice(M, 1, replace=False) 
    center_clusters = pixels[idx_random]

    # choose the next center_clusters 
    for j in range(K_clusters-1):
        distanceList = []
        for i in range(pixels.shape[0]):
            # pixel_clusters (each pixel belongs to cluster i) : shortest distances
            distance = sum(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            distanceList.append(distance)

        max_distance = max(distanceList)
        idx_random = np.append(idx_random, distanceList.index(max_distance))
        center_clusters = pixels[idx_random]


    # update K center_clusters value
    new_center_clusters = np.zeros_like(center_clusters)

    while (times):

        for i in range(pixels.shape[0]):
            # pixel_clusters (each pixel belongs to cluster i) : shortest distances
            idx = np.argsort(np.linalg.norm(pixels[i] - center_clusters,axis=1))
            pixel_clusters[i] = idx[0]

        for j in range(K_clusters):
            # update K center_clusters value : comput all pixel in cluster i, and get means.
            clusters_class = np.where(pixel_clusters == j)[0]
            new_center_clusters[j] = np.sum(pixels[clusters_class], axis=0) / clusters_class.shape[0]

        times -= 1
        return pixel_clusters.reshape(height, weight)

def computing_convergence_times(img, pixels, K_clusters):

    height = img.shape[0]
    weight = img.shape[1]

    pixel_clusters1 = np.zeros((height * weight), dtype=int)
    pixel_clusters2 = np.zeros((height * weight), dtype=int)

    M  = pixels.shape[0]
     
    idx_random1 = np.random.choice(M, 5, replace=False)
    idx_random2 = np.random.choice(M, 1, replace=False) 

    center_clusters1 = pixels[idx_random1]
    center_clusters2 = pixels[idx_random1]

    for j in range(K_clusters-1):
        distanceList = []
        for i in range(pixels.shape[0]):

            distance = sum(np.linalg.norm(pixels[i] - center_clusters2,axis=1))
            distanceList.append(distance)

        max_distance = max(distanceList)
        idx_random2 = np.append(idx_random2, distanceList.index(max_distance))
        center_clusters2 = pixels[idx_random2]


    new_center_clusters1 = np.zeros_like(center_clusters1)
    new_center_clusters2 = np.zeros_like(center_clusters2)


    count1 = 0; count2 = 0
    while True:

        for i in range(pixels.shape[0]):

            idx1 = np.argsort(np.linalg.norm(pixels[i] - center_clusters1,axis=1))
            idx2 = np.argsort(np.linalg.norm(pixels[i] - center_clusters2,axis=1))

            pixel_clusters1[i] = idx1[0]
            pixel_clusters2[i] = idx2[0]

        for j in range(K_clusters):

            clusters_class1 = np.where(pixel_clusters1 == j)[0]
            clusters_class2 = np.where(pixel_clusters2 == j)[0]
            
            new_center_clusters1[j] = np.sum(pixels[clusters_class1], axis=0) / clusters_class1.shape[0]
            new_center_clusters2[j] = np.sum(pixels[clusters_class2], axis=0) / clusters_class2.shape[0]

        
        if (np.allclose(new_center_clusters1,center_clusters1) and np.allclose(new_center_clusters2,center_clusters2)):
            
            print("K-means convergencetimes : " + str(count1))
            print("K-means-plus convergencetimes : " + str(count2))

            return 
        
        else:
            if (not(np.allclose(new_center_clusters1,center_clusters1))):
                center_clusters1 = new_center_clusters1.copy()
                count1 += 1
            
            if (not(np.allclose(new_center_clusters2,center_clusters2))):
                center_clusters2 = new_center_clusters2.copy()
                count2 += 1

def draw_clusters_on_image(img, pixel_clusters):
    
    K_clusters = int(pixel_clusters.max()) + 1
    
    average_color = np.zeros((K_clusters, 3))
    
    cluster_count = np.zeros(K_clusters)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = pixel_clusters[i,j]
            cluster_count[p] += 1
            average_color[p, :] += img[i, j, :]

    for p in range(K_clusters):
        average_color[p,:] /= float(cluster_count[p])

    result_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = pixel_clusters[i,j]
            result_img[i,j,:] = average_color[p,:]

    return result_img

def meanshift_segmentation(img, pixels, bandwidth):
    
    # initialization
    height = img.shape[0]
    weight = img.shape[1]

    # pixel_clusters (each pixel belongs to cluster i)
    pixel_clusters = np.zeros(pixels.shape[0], dtype=int)

    # record_pixels (record_pixels selected and unselected pixels)
    record_pixels = np.ones([height * weight], dtype=int)

    cluster_means = np.empty((0, pixels.shape[1]))

    # if there are still unselected pixels
    while (np.sum(record_pixels) > 0):
        
        # randomly choose a pixel (unselected pixel)
        idx = np.where(record_pixels > 0)[0]
        idx_unselected = idx[np.random.choice(idx.shape[0], 1)]
        cluster_mean = pixels[idx_unselected].flatten()

        # determine whether it is within the bandwidth range
        is_within = True

        while(is_within):

            distance = np.linalg.norm(pixels - cluster_mean, axis=1)
            idx_within = np.where(distance < bandwidth)[0]

            # update K center_clusters value
            new_cluster_mean = np.sum(pixels[idx_within], axis=0) / idx_within.shape[0]

            if np.linalg.norm(new_cluster_mean - cluster_mean) < bandwidth:  #norm 相減 平方 sum 再開根號
                is_within = False

            else:
                cluster_mean = new_cluster_mean.copy()

            # record the selected pixels as 0
            record_pixels[idx_within] = 0
            
        mean_distance = np.linalg.norm(cluster_means - new_cluster_mean,axis=1)


        if mean_distance.size > 0 and mean_distance[np.argsort(mean_distance)[0]] < bandwidth / 2:
            pixel_clusters[idx_within] = np.argsort(mean_distance)[0]

        else:
            cluster_means = np.vstack((cluster_means,new_cluster_mean))
            pixel_clusters[idx_within] = cluster_means.shape[0] 

    return pixel_clusters.reshape(height, weight)
   
def uniform_kernel_rgbspace(img, image):
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

    plt.savefig('uniform-kernel %s' %(image)) 

if __name__ == '__main__':

    K_clusters = [5, 7, 9]
    bandwidths = [0.3, 0.4, 0.5]

    # image = '2-image.jpg'
    image = '2-masterpiece.jpg'

    img = imageio.imread(image)

    pixels = np.zeros((img.shape[0] * img.shape[1], 5))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixels[i*img.shape[1] + j, :] = np.array([i, j,img[i, j, 0], img[i, j, 1], img[i, j, 2]])

    pixels_normalized = pixels / pixels.max(axis = 0)

    # Segmentation using K-Means、K-Means-plus
    for k in K_clusters:
        
        K_clustered_pixels = k_means_segmentation(img, pixels_normalized, k, times=50)
        Kplus_clustered_pixels = k_means_plus_segmentation(img, pixels_normalized, k, times=50)
        
        K_cluster_img = draw_clusters_on_image(img, K_clustered_pixels)
        Kplus_cluster_img = draw_clusters_on_image(img, Kplus_clustered_pixels)
        
        imageio.imsave('K-means with %d clusters on %s' % (int(k), image), K_cluster_img)
        imageio.imsave('K-means-plus with %d clusters on %s' % (int(k), image),  Kplus_cluster_img)

        # computing_convergence_times(img, pixels_normalized, k)

    # Segmentation using Meanshift
    for bandwidth in bandwidths:

        Meanshift_clustered_pixels = meanshift_segmentation(img, pixels_normalized, bandwidth)

        Meanshift_cluster_img = draw_clusters_on_image(img, Meanshift_clustered_pixels)

        imageio.imsave('Meanshift with bandwidth %.1f on %s' % (bandwidth, image), Meanshift_cluster_img)

        
        # Uniform Kernel on the RGB color space
        uniform_kernel_rgbspace(img, image)

        image_rgbspace = str('Meanshift with bandwidth %.1f on %s' % (bandwidth, image))
        img_rgbspace = imageio.imread('Meanshift with bandwidth %.1f on %s' % (bandwidth, image))

        uniform_kernel_rgbspace(img_rgbspace, image_rgbspace)






