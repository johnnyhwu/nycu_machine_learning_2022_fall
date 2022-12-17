import argparse
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from utils import compute_kernel


def aaaspectral_clustering(num_of_rows: int, num_of_cols: int, num_of_clusters: int, matrix_u: np.ndarray, mode: int,
                        cut: int, index: int) -> None:
    """
    Spectral clustering
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param mode: strategy for choosing centers
    :param cut: cut type
    :param index: index of the images
    :return: None
    """
    # Find initial centers
    info_log('=== Find initial centers of each cluster ===')
    centers = initial_centers(num_of_rows, num_of_cols, num_of_clusters, matrix_u, mode)

    # K-means
    info_log('=== K-means ===')
    clusters = kmeans(num_of_rows, num_of_cols, num_of_clusters, matrix_u, centers, index, mode, cut)

    # Plot data points in eigenspace if number of clusters is 2
    if num_of_clusters == 2:
        plot_the_result(matrix_u, clusters, index, mode, cut)


# def compute_matrix_u(matrix_w: np.ndarray, cut: int, num_of_clusters: int) -> np.ndarray:
#     """
#     Compute matrix U containing eigenvectors
#     :param matrix_w: weight matrix W
#     :param cut: cut type
#     :param num_of_clusters: number of clusters
#     :return: matrix U containing eigenvectors
#     """
#     # Get Laplacian matrix L and degree matrix D
#     matrix_d = np.zeros_like(matrix_w)
#     for idx, row in enumerate(matrix_w):
#         matrix_d[idx, idx] += np.sum(row)
#     matrix_l = matrix_d - matrix_w

#     if cut:
#         # Normalized cut
#         # Compute normalized Laplacian
#         for idx in range(len(matrix_d)):
#             matrix_d[idx, idx] = 1.0 / np.sqrt(matrix_d[idx, idx])
#         matrix_l = matrix_d.dot(matrix_l).dot(matrix_d)
#     # else is Ratio cut

#     # Get eigenvalues and eigenvectors
#     eigenvalues, eigenvectors = np.linalg.eig(matrix_l)
#     eigenvectors = eigenvectors.T

#     # Sort eigenvalues and find indices of nonzero eigenvalues
#     sort_idx = np.argsort(eigenvalues)
#     sort_idx = sort_idx[eigenvalues[sort_idx] > 0]

#     return eigenvectors[sort_idx[:num_of_clusters]].T


def initial_centers(num_of_rows: int, num_of_cols: int, num_of_clusters: int, matrix_u: np.ndarray,
                    mode: int) -> np.ndarray:
    """
    Get initial centers based on the given mode strategy
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param mode: strategy for choosing centers
    :return: initial centers
    """
    if not mode:
        # Random strategy
        return matrix_u[np.random.choice(num_of_rows * num_of_cols, num_of_clusters)]
    else:
        # k-means++ strategy
        # Construct indices of a grid
        grid = np.indices((num_of_rows, num_of_cols))
        row_indices = grid[0]
        col_indices = grid[1]

        # Construct indices vector
        indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))

        # Randomly pick first center
        num_of_points = num_of_rows * num_of_cols
        centers = [indices[np.random.choice(num_of_points, 1)[0]].tolist()]

        # Find remaining centers
        for _ in range(num_of_clusters - 1):
            # Compute minimum distance for each point from all found centers
            distance = np.zeros(num_of_points)
            for idx, point in enumerate(indices):
                min_distance = np.Inf
                for cen in centers:
                    dist = np.linalg.norm(point - cen)
                    min_distance = dist if dist < min_distance else min_distance
                distance[idx] = min_distance
            # Divide the distance by its sum to get probability
            distance /= np.sum(distance)
            # Get a new center
            centers.append(indices[np.random.choice(num_of_points, 1, p=distance)[0]].tolist())

        # Change from index to feature index
        for idx, cen in enumerate(centers):
            centers[idx] = matrix_u[cen[0] * num_of_rows + cen[1], :]

        return np.array(centers)


def kmeans(num_of_rows: int, num_of_cols: int, num_of_clusters: int, matrix_u: np.ndarray, centers: np.ndarray,
           index: int, mode: int, cut: int) -> np.ndarray:
    """
    K-means
    :param num_of_rows: number of rows
    :param num_of_cols: number of columns
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param centers: initial centers
    :param index: index of the images
    :param mode: strategy for choosing centers
    :param cut: cut type
    :return: cluster result
    """
    # Colors
    colors = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255]])
    if num_of_clusters > 3:
        colors = np.append(colors, np.random.choice(256, (num_of_clusters - 3, 3)), axis=0)

    # List storing images of clustering state
    num_of_points = num_of_rows * num_of_cols
    img = []

    # K-means
    current_centers = centers.copy()
    new_cluster = np.zeros(num_of_points, dtype=int)
    count = 0
    iteration = 100
    while True:
        # Display progress
        progress_log(count, iteration)

        # Get new cluster
        new_cluster = kmeans_clustering(num_of_points, num_of_clusters, matrix_u, current_centers)

        # Get new centers
        new_centers = kmeans_recompute_centers(num_of_clusters, matrix_u, new_cluster)

        # Capture new state
        img.append(capture_current_state(num_of_rows, num_of_cols, new_cluster, colors))

        if np.linalg.norm((new_centers - current_centers), ord=2) < 0.01 or count >= iteration:
            break

        # Update current parameters
        current_centers = new_centers.copy()
        count += 1

    # Save gif
    print()
    filename = f'./output/spectral_clustering/spectral_clustering_{index}_' \
               f'cluster{num_of_clusters}_' \
               f'{"kmeans++" if mode else "random"}_' \
               f'{"normalized" if cut else "ratio"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if len(img) > 1:
        img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)
    else:
        img[0].save(filename)

    return new_cluster


def kmeans_clustering(num_of_points: int, num_of_clusters: int, matrix_u: np.ndarray,
                      centers: np.ndarray) -> np.ndarray:
    """
    Classify data points into clusters
    :param num_of_points: number of data points
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param centers: current centers
    :return: cluster of each data point
    """
    new_cluster = np.zeros(num_of_points, dtype=int)
    for p in range(num_of_points):
        # Find minimum distance from data point to centers
        distance = np.zeros(num_of_clusters)
        for idx, cen in enumerate(centers):
            distance[idx] = np.linalg.norm((matrix_u[p] - cen), ord=2)
        # Classify data point into cluster according to the closest center
        new_cluster[p] = np.argmin(distance)

    return new_cluster


def kmeans_recompute_centers(num_of_clusters: int, matrix_u: np.ndarray, current_cluster: np.ndarray) -> np.ndarray:
    """
    Recompute centers according to current cluster
    :param num_of_clusters: number of clusters
    :param matrix_u: matrix U containing eigenvectors
    :param current_cluster: current cluster
    :return: new centers
    """
    new_centers = []
    for c in range(num_of_clusters):
        points_in_c = matrix_u[current_cluster == c]
        new_center = np.average(points_in_c, axis=0)
        new_centers.append(new_center)

    return np.array(new_centers)


def plot_the_result(matrix_u: np.ndarray, clusters: np.ndarray, index: int, mode: int, cut: int) -> None:
    """
    Plot data points in the eigenspace
    :param matrix_u: matrix U containing eigenvectors
    :param clusters: cluster result
    :param index: index of the image
    :param mode: strategy for choosing centers
    :param cut: cut type
    :return: None
    """
    colors = ['r', 'b']
    plt.clf()

    for idx, point in enumerate(matrix_u):
        plt.scatter(point[0], point[1], c=colors[clusters[idx]])

    # Save the figure
    filename = f'./output/spectral_clustering/eigenspace_{index}_' \
               f'{"kmeans++" if mode else "random"}_' \
               f'{"normalized" if cut else "ratio"}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)


def compute_laplacian(matrix_w: np.ndarray) -> np.ndarray:

    # compute matrix D
    matrix_d = np.zeros_like(matrix_w)
    for idx, row in enumerate(matrix_w):
        matrix_d[idx, idx] += np.sum(row)
    matrix_l = matrix_d - matrix_w
    
    # compute matrix L (Laplacian)
    matrix_l = matrix_d - matrix_w

    return matrix_l, matrix_d


def spectral_clustering(
    image: np.ndarray,
    gamma_spatial: float,
    gamma_color: float,
    num_cluster: int,
    init_cluster: str,
    cut_way: str,
    output_name: str
) -> None:

    # compute gram matrix
    matrix_w = compute_kernel(image, gamma_spatial, gamma_color)

    # compute laplacian matrix
    matrix_l, matrix_d = compute_laplacian(matrix_w)

    # normalize laplacian matrix if normalized cut
    if cut_way == "normalized":
        for r in range(matrix_d.shape[0]):
            matrix_d[r][r] = matrix_d[r][r] ** (-0.5)
        matrix_l = matrix_d @ matrix_l @ matrix_d

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

    with open("image1_ratio_matrix_u.npy", "wb") as f:
        np.save(f, matrix_u)
    exit()

    # normalize each row in matrix U if normalized cut
    if cut_way == "normalized":
        row_sum = np.sum(matrix_u, axis=1)
        for row_idx in range(matrix_u.shape[0]):
            matrix_u[row_idx, :] /= row_sum[row_idx]


if __name__ == '__main__':

    # user-defined variable
    IMAGE = "1"
    GAMMA_SPATIAL = 0.0001
    GAMMA_COLOR = 0.001
    NUM_CLUSTER = 3
    INIT_CLUSTER = "k-means++" # "random" or "k-means++"
    CUT_WAY = "ratio" # "ratio" or "normalized"

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
        f"image{IMAGE}_{INIT_CLUSTER}_{NUM_CLUSTER}"
    )

    exit()

    for i, im in enumerate(images):

        # Spectral clustering
        info_log('=== Spectral clustering ===')
        spectral_clustering(rows, columns, clu, m_u, m, cu, i)