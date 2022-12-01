import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# SIFT interest point detection and matching
def sift_detection_and_matching(image1, image2, point, threshold):

    # img1 = cv2.imread(image1)
    # img2 = cv2.imread(image2)
    img1 = plt.imread(image1)
    img2 = plt.imread(image2)
    # SIFT interest point detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    # SIFT feature matching
    match1=[]
    dis = np.zeros(np.shape(des1))
    for i in range(des1.shape[0]):
        if np.std(des1[i,:]) != 0:
            dis = des2-des1[i,:]
            dis = np.linalg.norm(dis, axis=1)
            arrange = np.argsort(dis).tolist()
            if dis[arrange[0]]/dis[arrange[1]] <= threshold:
                match1.append((i,arrange[0]))

    match2=[]
    for i in range(des2.shape[0]):
        if np.std(des2[i,:]) != 0:
            dis=des1-des2[i,:]
            dis=np.linalg.norm(dis, axis=1)
            arrange =np.argsort(dis).tolist()
            if dis[arrange[0]]/dis[arrange[1]] <= threshold:
                match2.append((arrange[0],i))
            
    match = list(set(match1).intersection(set(match2)))
    matches = [cv2.DMatch(i[0], i[1], 1) for i in match]

    # Plot the detected interest points and point correspondences
    draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (0,165,255),
                        flags = 0)
    
    matching = cv2.drawMatches(img1, kp1, img2, kp2, matches[:point], None,**draw_params)
    cv2.imwrite("matching3.jpg", matching)




    n = des1.shape[0]
    matches = np.empty((0,2), int)
    for i in range(n):
        feature = des1[i]
        distances = np.linalg.norm(des2 - feature,axis=1)
        idx = np.argsort(distances)
        
        if distances[idx[0]] < threshold * distances[idx[1]]:
            match = np.array([i,idx[0]]).reshape(1,2)
            matches = np.vstack((matches,match))
        
    
    inliers, model = refine_match(kp1, kp2, matches)
    # plot_matches(img1, img2, keypoints1, keypoints2, matches[inliers,:])
    # ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches[inliers,:], None,**draw_params)

    keypointsList1 = []
    for i in range(len(kp1)):
        keypointsList1.append(kp1[i].pt)
    

    keypointsList1 = np.array(keypointsList1)

    keypointsList2 = []
    for i in range(len(kp2)):
        keypointsList2.append(kp2[i].pt)
    

    keypointsList2 = np.array(keypointsList2)


    fig = plt.figure()
    new_im = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], max(img1.shape[2], img2.shape[2])),dtype=np.uint8)
    new_im[:img1.shape[0], :img1.shape[1], :img1.shape[2]] = img1
    new_im[:img2.shape[0], img1.shape[1]:, :img2.shape[2]] = img2
    plt.imshow(new_im)
    plt.autoscale(False)


    for m in matches:
        ind1, ind2 = m
        plt.plot([keypointsList1[ind1,0], img1.shape[1]+keypointsList2[ind2,0]], [keypointsList1[ind1,1], keypointsList2[ind2,1]])
    plt.savefig('plot.png') 
    plt.show()

    # cv2.imwrite("ransac.jpg", ransac)

    
    # matches= np.array(matches)

    print(matches)


def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10, num_iterations = 1000):

    n = matches.shape[0]
    seq = [i for i in range(n)]
    valid_sample = [i for i in range(4,n)]
    
    inliers = np.array([])
    H_best = np.zeros((3,3))
    
    keypointsList1 = []
    for i in range(len(keypoints1)):
        keypointsList1.append(keypoints1[i].pt)
    

    keypointsList1 = np.array(keypointsList1)

    keypointsList2 = []
    for i in range(len(keypoints2)):
        keypointsList2.append(keypoints2[i].pt)
    

    keypointsList2 = np.array(keypointsList2)

    pts_homo = np.ones((keypointsList1.shape[0],3))
    pts_homo[:,:-1] = keypointsList1[:,:2]
    for _ in range(num_iterations):
        
        # valid sample subset 
        sample_length = random.sample(valid_sample, 1)[0]
        sample = random.sample(seq,sample_length)
        
        candit = matches[sample]
        # 1st image coordinates
        pts = pts_homo[candit[:,0]]
        pts_prime = keypointsList2[candit[:,1],:2]
        
        A = np.zeros((sample_length*2,9))
        
        # homography
        for i in range(sample_length):
            idx = 2 * i
            A[idx,3:6]  = -pts[i]
            A[idx,6:]   = pts[i] * pts_prime[i,1]
            A[idx+1,:3] = pts[i]
            A[idx+1,6:] = -pts[i] * pts_prime[i,0]
        
        u,s,vh = np.linalg.svd(A)
        H = vh[-1].reshape(3,3)
        
        # reprojection error
        pts_reproj_homo = H.dot(pts_homo[matches[:,0]].T)
        pts_reproj_inhomo = pts_reproj_homo[:-1,:] / pts_reproj_homo[-1,:]
        error = np.linalg.norm(keypointsList2[matches[:,1],:2]-pts_reproj_inhomo.T,axis=1)
        inlier_idx = np.where(error < reprojection_threshold)[0]
        
        if len(inliers) < len(inlier_idx):
            inliers = inlier_idx
            H_best = H
        
    return inliers, H_best

if __name__ == '__main__': 
    
    image = "1-image.jpg"
    image1 = "1-book1.jpg"
    image2 = "1-book2.jpg"
    image3 = "1-book3.jpg"

    sift_detection_and_matching(image1, image, point=1000, threshold=0.6)
    sift_detection_and_matching(image2, image, point=1000, threshold=0.5)
    sift_detection_and_matching(image3, image, point=1000, threshold=0.9)

    
