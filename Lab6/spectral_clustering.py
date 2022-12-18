import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import compute_kernel, cluster2image


def compute_laplacian(matrix_w: np.ndarray) -> np.ndarray:

    # compute matrix D
    matrix_d = np.zeros_like(matrix_w)
    for idx, row in enumerate(matrix_w):
        matrix_d[idx, idx] += np.sum(row)
    matrix_l = matrix_d - matrix_w
    
    # compute matrix L (Laplacian)
    matrix_l = matrix_d - matrix_w

    return matrix_l, matrix_d


def find_initial_center(num_cluster: int, matrix_u: np.ndarray, init_cluster: str) -> np.ndarray:
    
    if init_cluster == "random":
        idx = np.random.choice(10000, num_cluster)
        return matrix_u[idx]
    else:
        cluster_center = []

        # spatial information
        coordinate = []
        for row_pos in range(100):
            for col_pos in range(100):
                coordinate.append([row_pos, col_pos])

        # choose first center randomly
        idx = np.random.choice(a=10000, size=1)[0]
        cluster_center.append(coordinate[idx])

        # find remaining centers
        for _ in range(num_cluster-1):
            
            # distance of each data point to its nearest cluster
            distance = np.zeros(10000)

            for idx, coord in enumerate(coordinate):

                min_distance = np.Inf
                for center in cluster_center:
                    dist = np.linalg.norm(np.array(coord) - np.array(center))
                    if dist < min_distance:
                        min_distance = dist

                distance[idx] = min_distance

            # convert distance into probability
            distance /= np.sum(distance)

            # choose the data point with higher probability as new center of
            # the data point is far away from all of existing centers
            idx = np.random.choice(10000, 1, p=distance)[0]
            cluster_center.append(coordinate[idx])

        # find the corresponding eigenvectors given coordinates of cluster's center
        matrix_u_part = np.zeros((len(cluster_center), matrix_u.shape[1]))
        for idx, center in enumerate(cluster_center):
            center_idx = center[0] * 100 + center[1]
            matrix_u_part[idx][:] = matrix_u[center_idx][:]

        return matrix_u_part


def clustering(num_cluster: int, matrix_u: np.ndarray, init_cluster: str, output_name: str) -> None:

    # find initial center of clusters
    cluster_center = find_initial_center(num_cluster, matrix_u, init_cluster)

    # color of each cluster
    cluster_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for _ in range(num_cluster - 3):
        new_color = np.random.choice(256, (3, )).tolist()
        cluster_color.append(new_color)
    
    # save a list of cluster images (will be converted into gif)
    cluster_image = []

    # spectral clustering
    current_center = cluster_center.copy()
    current_cluster = np.zeros(10000, dtype=np.int32)

    for iter_idx in range(100):

        print(f"Iter: {iter_idx+1:02d}")

        # caculate current cluster
        for point_idx in range(10000):
            
            # compute the distance of current data point to all clusters
            distance = np.zeros(num_cluster)
            for idx, center in enumerate(current_center):
                distance[idx] = np.linalg.norm((matrix_u[point_idx] - center), ord=2)
            
            # current data point belongs to the nearest cluster
            current_cluster[point_idx] = np.argmin(distance)

        # save current cluster as image
        cluster_image.append(cluster2image(current_cluster, cluster_color))

        # update each cluster's center
        new_center = np.zeros_like(current_center)
        for cluster_idx in range(num_cluster):
            mask = (current_cluster == cluster_idx)
            cluster_point = matrix_u[mask]
            cluster_center = np.average(cluster_point, axis=0)
            new_center[cluster_idx][:] = cluster_center[:]
        
        # converge
        if np.linalg.norm((new_center - current_center), ord=2) < 0.01:
            break

        current_center = new_center.copy()
        

    # save list of images as a gif
    print("save gif")
    if len(cluster_image) == 1:
        cluster_image[0].save(f"result/spectral/{output_name}.gif")
    else:
        cluster_image[0].save(
            f"result/spectral/{output_name}.gif",
            save_all=True,
            append_images=cluster_image[1:],
            duration=300,
            loop=0
        )
    
    cluster_image[0].save(f"result/spectral/{output_name}_first.png")
    cluster_image[-1].save(f"result/spectral/{output_name}_last.png")

    # plot data point in eigenspace (only for 2 clusters case)
    print("save png")
    if num_cluster == 2:
        for idx, point in enumerate(matrix_u):
            if current_cluster[idx] == 0:
                plt.scatter(point[0], point[1], c="red")
            else:
                plt.scatter(point[0], point[1], c="green")
        plt.savefig(f"result/spectral/{output_name}.png")
    

def spectral_clustering(
    image: np.ndarray,
    gamma_spatial: float,
    gamma_color: float,
    num_cluster: int,
    init_cluster: str,
    cut_way: str,
    output_name: str
) -> None:

    # shortcut
    # matrix_u = np.load("image_2_normalized_matrix_u.npy")
    # print(matrix_u.shape)
    # clustering(num_cluster, matrix_u, init_cluster, output_name)
    # exit()

    # compute gram matrix
    matrix_w = compute_kernel(image, gamma_spatial, gamma_color)

    # compute laplacian matrix
    matrix_l, matrix_d = compute_laplacian(matrix_w)

    # normalize laplacian matrix if normalized cut
    if cut_way == "normalized":
        for r in range(matrix_d.shape[0]):
            matrix_d[r][r] = 1.0 / np.sqrt(matrix_d[r, r])
        matrix_l = np.matmul(np.matmul(matrix_d, matrix_l), matrix_d)

    # calculate eigenvalues and eigenvectors of laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix_l)
    eigenvectors = eigenvectors.T

    # find non-zero eigenvalues and sort it
    sort_idx = np.argsort(eigenvalues)
    mask = eigenvalues[sort_idx] > 0
    sort_idx = sort_idx[mask]

    # only keep first k eigenvectors
    sort_idx = sort_idx[:num_cluster]

    # get corresponding eigenvectors
    matrix_u = eigenvectors[sort_idx].T

    # normalize each row in matrix U if normalized cut
    if cut_way == "normalized":
        for row_idx in range(matrix_u.shape[0]):
            matrix_u[row_idx, :] /= np.sum(matrix_u[row_idx, :])
    
    # save matrix U for avoid recomputing it
    # with open("image_2_normalized_matrix_u.npy", "wb") as f:
    #     np.save(f, matrix_u)

    # spectral clustering
    clustering(num_cluster, matrix_u, init_cluster, output_name)


if __name__ == '__main__':

    os.makedirs("result/spectral", exist_ok=True)

    # user-defined variable
    IMAGE = "2"
    GAMMA_SPATIAL = 0.0001
    GAMMA_COLOR = 0.0001
    NUM_CLUSTER = 3
    INIT_CLUSTER = "k-means++" # "random" or "k-means++"
    CUT_WAY = "normalized" # "ratio" or "normalized"

    # read image as numpy array
    image = np.asarray(Image.open(f"img/image{IMAGE}.png"))

    assert NUM_CLUSTER >= 2
    assert INIT_CLUSTER == "random" or INIT_CLUSTER == "k-means++"
    assert CUT_WAY == "ratio" or CUT_WAY == "normalized"
    assert image.shape[0] == image.shape[1] == 100

    # spectral clustering
    spectral_clustering(
        image,
        GAMMA_SPATIAL,
        GAMMA_COLOR,
        NUM_CLUSTER,
        INIT_CLUSTER,
        CUT_WAY,
        f"image{IMAGE}_{INIT_CLUSTER}_{NUM_CLUSTER}_{CUT_WAY}"
    )