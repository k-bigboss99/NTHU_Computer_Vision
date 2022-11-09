import numpy as np
import skimage.color  
import skimage.io
from matplotlib import pyplot as plt


# Compute Fundamental Matrix
def compute_fundamental_matrix(point1, point2):

    A = np.zeros((point1.shape[0], 9))
    
    for i in range(point1.shape[0]):
        u1 = point1[i][0]
        v1 = point1[i][1]
        u2 = point2[i][0]
        v2 = point2[i][1]

        A[i] = np.array([u1*u2, u2*v1, u2, v2*u1, v1*v2, v2, u1, v1, 1])

    U, S, V = np.linalg.svd(A, full_matrices=True)
    f = V[-1, :]
    Fundamental_matrix = f.reshape(3, 3)

    # constrain Fundamental_matrix : make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(Fundamental_matrix, full_matrices=True)
    S[-1] = 0 
    Fundamental_matrix = np.dot(U, np.dot(np.diag(S), V))
    
    return Fundamental_matrix 

# Compute Fundamental Matrix Normalized
def compute_fundamental_matrix_normalized(point1, point2):
    N = point1.shape[0]

    points1_uv = point1[:, 0:2]
    points2_uv = point2[:, 0:2]

    mean1 = np.mean(points1_uv, axis=0)
    mean2 = np.mean(points2_uv, axis=0)

    points1_center = points1_uv - mean1
    points2_center = points2_uv - mean2

    scale1 = np.sqrt(2/(np.sum(points1_center**2)/N * 1.0))
    scale2 = np.sqrt(2/(np.sum(points2_center**2)/N * 1.0))

    T1 = np.array([[scale1,      0, -mean1[0] * scale1],
                   [     0, scale1, -mean1[1] * scale1],
                   [     0,      0,                  1]])

    T2 = np.array([[scale2,      0, -mean2[0] * scale2],
                   [     0, scale2, -mean2[0] * scale2],
                   [     0,      0,                  1]])

    point1_normalize = np.dot(T1, np.transpose(point1))
    point2_normalize = np.dot(T2, np.transpose(point2))

    Fundamental_normalize = compute_fundamental_matrix(np.transpose(point1_normalize), np.transpose(point2_normalize))
    Normalized_fundamental_matrix = np.dot(np.transpose(T2), np.dot(Fundamental_normalize, T1))
    
    return Normalized_fundamental_matrix

# Plot Epipolar Lines
def plot_epipolar_lines(Fundamental_matrix, img1, img2, point1, point2, name):
    
    def plot_line(coeffs, xlim):

        a, b, c = coeffs
        x = np.linspace(xlim[0], xlim[1], 100)
        y = (a * x + c) / -b
        return x, y

    def compute_epipole(Fundamental_matrix):
        U, S, V = np.linalg.svd(Fundamental_matrix)
        epipole = V[-1, :]
        epipole = epipole / epipole[2]
        return epipole

    height = img1.shape[0]
    weight = img1.shape[1]
    nrows = 2; ncols = 1
    show_epipole = False
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))

    ax1 = axes[0]
    ax1.set_title("Image 1")
    ax1.imshow(img1, cmap="gray")

    ax2 = axes[1]
    ax2.set_title("Image 2")
    ax2.imshow(img2, cmap="gray")
    
    for i in range(point1.shape[0]):
        p1 = np.transpose(point1)[:, i]
        p2 = np.transpose(point2)[:, i]
        
        # Epipolar line in the image of camera 1 given the point in the image of camera 2
        coeffs = np.dot(np.transpose(p2), Fundamental_matrix)
        x, y = plot_line(coeffs, (-1500, weight))
        ax1.plot(x, y, color="red")
        ax1.scatter(*p1.reshape(-1)[:2], color="blue")

        # Epipolar line in the image of camera 2 given the point in the image of camera 1
        coeffs = np.dot(Fundamental_matrix, p1)
        x, y = plot_line(coeffs, (0, 2800))
        ax2.plot(x, y, color="red")
        ax2.scatter(*p2.reshape(-1)[:2], color="blue")
        
    if show_epipole:
        e1 = compute_epipole(Fundamental_matrix)
        e2 = compute_epipole(np.transpose(Fundamental_matrix))

        ax1.scatter(*e1.reshape(-1)[:2], color="red")
        ax2.scatter(*e2.reshape(-1)[:2], color="red")
    
    else:
        ax1.set_xlim(0, weight)
        ax1.set_ylim(height, 0)
        ax2.set_xlim(0, weight)
        ax2.set_ylim(height, 0)

    plt.tight_layout()

    plt.savefig(name + '_img.jpg')

# Compute distance to epipolar line
def compute_distance_to_epipolar_line(point1,point2,Fundamental_matrix):
    l = np.dot(np.transpose(Fundamental_matrix), (np.transpose(point2)))
   
    dist_sum = 0
    for i in range(point1.shape[0]):
        dist_sum += np.abs(point1[i][0]*l[0][i] + point1[i][1]*l[1][i] + l[2][i]) / np.sqrt(l[0][i]**2 + l[1][i]**2) 
    return dist_sum / (point1.shape[0])


if __name__ == '__main__': 
    
    img1 = skimage.io.imread("image1.jpg")
    img2 = skimage.io.imread("image2.jpg")

    point1 = np.loadtxt("pt_2D_1.txt", delimiter=' ', skiprows=1, usecols=(0,1)) 
    point2 = np.loadtxt("pt_2D_2.txt", delimiter=' ', skiprows=1, usecols=(0,1))

    point1 = np.column_stack([point1, [1]*46])
    point2 = np.column_stack([point2, [1]*46])

  
    Fundamental_matrix = compute_fundamental_matrix(point1, point2)
    print(Fundamental_matrix)
    plot_epipolar_lines(Fundamental_matrix, img1, img2, point1, point2, name='a')

    F_normalized = compute_fundamental_matrix_normalized(point1, point2)
    print(F_normalized)
    plot_epipolar_lines(F_normalized, img1, img2, point1, point2, name='b')
    
    dist = compute_distance_to_epipolar_line(point1,point2,F_normalized)
    print("distance to epipolar line:", dist)