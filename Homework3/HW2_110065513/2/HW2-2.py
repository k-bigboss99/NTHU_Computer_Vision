import numpy as np
import cv2

def draw(image, points):
    """
    given 4 points in fig
    draw 4 points and 4 lines
    """
    img = image.copy()
    p_mat = np.append(points, points[0,:].reshape(1, -1), axis=0)
    
    for i in range(4):
        cv2.circle(img,(p_mat[i, 1], p_mat[i, 0]), 1, (255, 0, 0), -1)
        cv2.line(img, (p_mat[i, 1], p_mat[i, 0]), (p_mat[i+1, 1], p_mat[i+1, 0]), (0, 0, 255), 5)
    cv2.imwrite('./output/selected_img.jpg', img)
    return img

        
def calc_H(src_points, tgt_points):
    num_point, _ = src_points.shape

    Z = np.zeros((2*num_point, 9)) 
    for i in range(num_point):   
        Z[i*2, 0:3] = src_points[i, :]
        Z[i*2, 6:] = src_points[i, :] * (-tgt_points[i, 0])
        Z[i*2+1, 3:6] = src_points[i, :]
        Z[i*2+1, 6:] = src_points[i, :]* (-tgt_points[i, 1])
        
    s, v = np.linalg.eig(np.transpose(Z)@Z) 
    idx = np.argsort(s)
    H = v[:, idx[0]].reshape((3,3))   

    # scaling
    H *= 1./np.linalg.norm(H)
    
    return H

def bilinear_interpolation(corr_points, src_img):
    """
    corr_points with float type
    find the weighted intensity (distance with 4 neighbors)
    """
    upper = np.ceil(corr_points)
    lower = np.floor(corr_points)
    upper_diff = upper - corr_points
    lower_diff = corr_points - lower
    
    src_intensity = src_img[int(lower[0]):int(lower[0])+2, int(lower[1]):int(lower[1])+2, :]
    
    output = src_intensity[0, 0, :]*upper_diff[0]*upper_diff[1] + src_intensity[0, 1, :]*upper_diff[0]*lower_diff[1] + \
            src_intensity[1, 0, :]*upper_diff[1]*lower_diff[0] + src_intensity[1, 1, :]*lower_diff[0]*lower_diff[1]
    
    return output

if __name__ == '__main__': 
    # params.
    ori_img = cv2.imread("Delta-Building.jpg")
    src_points = np.array([[357, 436, 1],
                        [801, 422, 1],
                        [1016, 893, 1],
                        [108, 889, 1]])
    src_img = draw(ori_img, src_points)
    h, w, _ = ori_img.shape                    
    tgt_points = np.array([[0, 0, 1],
                        [h, 0, 1],
                        [h, w, 1],
                        [0, w, 1]])
    
    # backward
    H = calc_H(tgt_points, src_points)
    print('H:\n {}'.format(H))

    # display image
    tgt_img = np.zeros_like(ori_img, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            corr_point = np.dot(H, np.array([i, j, 1]).T)
            corr_point = corr_point / corr_point[2]
            
            intensity = bilinear_interpolation(corr_point[:2], ori_img)               
            tgt_img[i, j, :] = intensity
    cv2.imwrite('./output/rectified_img.jpg', tgt_img)