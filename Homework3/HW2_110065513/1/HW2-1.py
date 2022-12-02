import numpy as np
import cv2


def txt2mat(txt_path):
    """
    Transfer point.txt to point matrix in virtual img. plane
    output
        point with corresponding coor., shape: (number of total point, 3), 3=(u, v, 1)
    """
    with open(txt_path) as f:
        lines = f.readlines()
        point_mat = np.ones((int(lines[0]), 3))

        for i in np.arange(1, int(lines[0])+1):
            coor = lines[i].split(' ')
            point_mat[i-1, 0] = float(coor[0])
            point_mat[i-1, 1] = float(coor[1])

    return point_mat

def EightPoint_algo(p1_mat, p2_mat, scaling1=False, scaling2=False):
    """
    inputs : 2 2D points captured from 2 camreas
    output: foundermental mat. 
    """
    num_eq, _ = p1_mat.shape

    # Z_i = [X_i1 * X_i0.T]
    Z = np.zeros((num_eq, 9)) 
    for i in range(num_eq):   
        X_i0 = p1_mat[i, :].reshape(-1, 1) # (u, v, 1).T
        X_i1 = p2_mat[i, :].reshape(1, -1) # (u', v', 1)
        Z[i, :] = np.dot(X_i0, X_i1).reshape(1, -1)

    # solve ZF=0  
    s, v = np.linalg.eig(np.transpose(Z)@Z) 
    idx = np.argsort(s)
    least_eigenV = v[:,idx[0]].reshape((3,3)) 
    f = CheckRank2(least_eigenV, scaling1, scaling2)

    return f

def CheckRank2(f, scaling1=False, scaling2=False):
    """
    set scaling term=faise if the points aren't normalize
    if normalize, scaling matix with shape 3*3
    """
    fu, fs, fvh = np.linalg.svd(f)
    
    if isinstance(scaling1, bool):
        f = fu @ np.diag([*fs[:2], 0]) @ fvh
    else:
        f = scaling1.T @ fu @ np.diag([*fs[:2], 0]) @ fvh  @ scaling2

    # scaling 
    f *= 1./np.linalg.norm(f)
    
    return f



def Norm(p_mat):
    """
    inputs : 2D points captured from 2 camreas
    output: norm 2D points 
    """

    num_eq, _ = p_mat.shape

    # compute center
    p_mean = np.mean(p_mat, axis=0)
    p_center = p_mat - np.tile(p_mean, (num_eq, 1))

    # scaling by distance
    avg_dist = np.mean((p_center[:,0]**2 + p_center[:,1]**2)**(0.5))
    scaling = np.sqrt(2) / avg_dist
    norm_p = p_center * scaling
    norm_p = np.concatenate((norm_p, np.ones((num_eq, 1))), axis=1)
    
    scaling_mat = np.array([[scaling, 0, -scaling*p_mean[0]],
                            [0, scaling, -scaling*p_mean[1]],
                            [0, 0, 1]])

    return norm_p, scaling_mat

def calc_distance(line, point):
    """
    calculate distance between given line and point
    ax+by+c=0
    a: line[0], b:line[1]...
    """
    distance = np.abs(line[0]*point[0] + line[1]*point[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)   
    return distance

def draw_epilines(image1, image2, f, p1_mat, p2_mat):
    img1 = image1.copy()
    img2 = image2.copy()
    _, W1, _ = image1.shape
    _, W2, _ = image2.shape

    total_dist1 = 0
    total_dist2 = 0
    for i in range(p1_mat.shape[0]):
        # draw on l2
        # ax+by+c=0
        x2 = np.matmul(f.T, p1_mat[i, :].reshape(-1, 1))        
        left_p = (0, int(-x2[2]/x2[1]))
        right_p = (W2, int(-(x2[0]*W2 + x2[2])/x2[1]))
        total_dist2 += calc_distance(x2, p2_mat[i, :])
        cv2.line(img2, left_p, right_p, (0, 255, 255), 1)
        cv2.circle(img2, (int(p2_mat[i, 0]), int(p2_mat[i, 1])), 2, (0, 255, 0), 2)

        # draw on l1
        x1 = np.matmul(f, p2_mat[i, :].reshape(-1, 1))
        left_p = (0, int(-x1[2]/x1[1]))
        right_p = (W1, int(-(x1[0]*W1 + x1[2])/x1[1]))
        total_dist1 += calc_distance(x1, p1_mat[i, :])
        cv2.line(img1, left_p, right_p, (0, 255, 255), 1)
        cv2.circle(img1, (int(p1_mat[i, 0]), int(p1_mat[i, 1])), 2, (0, 255, 0), 2)
    avg_dist1 = total_dist1/p1_mat.shape[0]
    avg_dist2 = total_dist2/p1_mat.shape[0]

    return img1, img2, avg_dist1, avg_dist2



if __name__ == '__main__': 
    # params.
    image1 = cv2.imread("image1.jpg")
    image2 = cv2.imread("image2.jpg")

    #(a)
    point1 = txt2mat('pt_2D_1.txt')
    point2 = txt2mat('pt_2D_2.txt')
    f = EightPoint_algo(point1, point2)
    print('f in (a):\n {}\n'.format(f))

    #(b)
    p1_norm, scaling1 = Norm(point1[:,:2])
    p2_norm, scaling2 = Norm(point2[:,:2])
    f_norm = EightPoint_algo(p1_norm, p2_norm, scaling1, scaling2)
    print('f in (b):\n {}\n'.format(f_norm))

    #(c)
    # for f produced from (a)
    img1, img2, err1, err2 = draw_epilines(image1, image2, f, point1, point2)
    cv2.imwrite('./output/a_img1.jpg', img1)
    cv2.imwrite('./output/a_img2.jpg', img2)
    print('----Average distance between points and lines in (a)----')
    print('Fig.1: {}'.format(err1[0]))
    print('Fig.2: {}\n'.format(err2[0]))

    # for f produced from (b)
    img1, img2, err1, err2 = draw_epilines(image1, image2, f_norm, point1, point2)
    cv2.imwrite('./output/b_img1.jpg', img1)
    cv2.imwrite('./output/b_img2.jpg', img2)
    print('----Average distance between points and lines in (b)----')
    print('Fig.1: {}'.format(err1[0]))
    print('Fig.2: {}\n'.format(err2[0]))

