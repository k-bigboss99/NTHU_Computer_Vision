import cv2
import numpy as np

# SIFT interest point detection and matching
def sift_detection_and_matching(image1, image2, point):

    img1= cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # SIFT interest point detection
    sift = cv2.xfeatures2d.SIFT_create()
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
            if dis[arrange[0]]/dis[arrange[1]] <= 0.5:
                match1.append((i,arrange[0]))

    match2=[]
    for i in range(des2.shape[0]):
        if np.std(des2[i,:]) != 0:
            dis=des1-des2[i,:]
            dis=np.linalg.norm(dis, axis=1)
            arrange =np.argsort(dis).tolist()
            if dis[arrange[0]]/dis[arrange[1]] <= 0.5:
                match2.append((arrange[0],i))
            
    match = list(set(match1).intersection(set(match2)))
    matches = [cv2.DMatch(i[0], i[1], 1) for i in match]

    # Plot the detected interest points and point correspondences
    draw_params = dict(matchColor = (0,0,255),
                        singlePointColor = (255,0,0),
                        flags = 0)
    
    matching = cv2.drawMatches(img1, kp1, img2, kp2, matches[:point], None,**draw_params)
    cv2.imwrite("matching2.jpg", matching)

if __name__ == '__main__': 
    
    image1 = "1a_notredame.jpg"
    image2 = "1b_notredame.jpg"

    sift_detection_and_matching(image1, image2, point=100)