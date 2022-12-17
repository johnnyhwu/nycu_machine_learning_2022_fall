import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

def compute_kernel(image: np.ndarray, gamma_spatial: float, gamma_color: float) -> np.ndarray:
    
    # spatial similarity
    coordinate = []
    for row_pos in range(100):
        for col_pos in range(100):
            coordinate.append([row_pos, col_pos])
    coordinate = np.array(coordinate)
    spatial_similarity = cdist(coordinate, coordinate, "sqeuclidean")

    # color similarity
    color = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    color_similarity = cdist(color, color, 'sqeuclidean')

    # kernel function
    result = np.multiply(np.exp(-gamma_spatial * spatial_similarity), np.exp(-gamma_color * color_similarity))
    
    return result


def cluster2image(initial_cluster: np.ndarray, cluster_color: list) -> Image:
    image = np.zeros((100*100, 3))
    for pixel_idx in range(10000):
        pixel_cluster = initial_cluster[pixel_idx]
        pixel_color = cluster_color[pixel_cluster]
        image[pixel_idx, :] = pixel_color[:]
    image = np.reshape(image, (100, 100, 3))
    image = np.uint8(image)
    image = Image.fromarray(image)
    return image