import numpy as np
from PIL import Image
from utils import compute_kernel, cluster2image

def initialize_clustering(num_cluster: int, init_cluster: str, kernel: np.ndarray) -> np.ndarray:

    # choose center of cluster
    if init_cluster == "random":
        cluster_center = np.random.choice(100, (num_cluster, 2))
    elif init_cluster == "k-means++":
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
        
        cluster_center = np.array(cluster_center)
    

    # after we get centers of clusters, we can determine each data point
    initial_cluster = np.zeros(10000, dtype=np.int32)
    
    for point_idx in range(10000):
        
        # compute the distance of current data point to all clusters
        distance = np.zeros(num_cluster)
        for idx, center in enumerate(cluster_center):
            center_idx = center[0] * 100 + center[1]
            distance[idx] = kernel[point_idx, point_idx] + kernel[center_idx, center_idx] - 2 * kernel[point_idx, center_idx]
        
        # Pick the index of minimum distance as the cluster of the point
        initial_cluster[point_idx] = np.argmin(distance)
    
    return initial_cluster


def clustering(num_cluster: int, initial_cluster: np.ndarray, kernel: np.ndarray, output_name: str) -> None:

    # color of each cluster
    cluster_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for _ in range(num_cluster - 3):
        new_color = np.random.choice(256, (3, )).tolist()
        cluster_color.append(new_color)
    
    # save a list of cluster images (will be converted into gif)
    cluster_image = [cluster2image(initial_cluster, cluster_color)]

    # kernel k-means clustering
    current_cluster = initial_cluster.copy()

    for iter_idx in range(100):
        
        # calculate the size of each cluster
        cluster_size = np.zeros(num_cluster)
        for val in current_cluster:
            cluster_size[val] += 1
        
        # calculate sum of pairwise kernel distance of each cluster
        cluster_pairwise = np.zeros(num_cluster)
        for cluster_idx in range(num_cluster):
            tmp_kernel = kernel.copy()
            for point_idx in range(10000):
                
                # if the data point not in this cluster, zero out its kernel distance
                if current_cluster[point_idx] != cluster_idx:
                    tmp_kernel[point_idx, :] = 0
                    tmp_kernel[:, point_idx] = 0

            cluster_pairwise[cluster_idx] = np.sum(tmp_kernel)

        # avoid cluster size equals zero
        cluster_size += 1

        # marginalize pairwise distance
        cluster_pairwise /= cluster_size ** 2
        
        # compute distance of each data point to all cluster
        new_cluster = np.zeros(10000, dtype=np.int32)
        
        for point_idx in range(10000):
            distance = np.zeros(num_cluster)
            for cluster_idx in range(num_cluster):
                first_item = kernel[point_idx, point_idx]
                mask = np.where(current_cluster == cluster_idx)
                second_item = np.sum(kernel[point_idx, mask])
                second_item *= (2.0 / cluster_size[cluster_idx])
                third_item = cluster_pairwise[cluster_idx]
                distance[cluster_idx] = first_item - second_item + third_item
            
            new_cluster[point_idx] = np.argmin(distance)
        
        # save cluster as image
        cluster_image.append(cluster2image(new_cluster, cluster_color))

        # converge
        if np.linalg.norm((new_cluster - current_cluster)) < 0.01:
            break

        current_cluster = new_cluster.copy()
        print(f"Iter: {iter_idx+1:02d}", end="\r")

    # save list of images as a gif
    cluster_image[0].save(
        f"result/kernel_kmeans/{output_name}.gif",
        save_all=True,
        append_images=cluster_image[1:],
        duration=150,
        loop=0
    )

    cluster_image[0].save(f"result/kernel_kmeans/{output_name}_first.png")
    cluster_image[-1].save(f"result/kernel_kmeans/{output_name}_last.png")



def kernel_kmeans(
    image: np.ndarray,
    gamma_spatial: float,
    gamma_color: float,
    num_cluster: int,
    init_cluster: str,
    output_name: str
) -> None:

    # compute gram matrix
    kernel = compute_kernel(image, gamma_spatial, gamma_color)

    # clustering initialization
    initial_cluster = initialize_clustering(num_cluster, init_cluster, kernel)

    # clustering
    clustering(num_cluster, initial_cluster, kernel, output_name)


if __name__ == '__main__':

    # user-defined variable
    IMAGE = "2"
    GAMMA_SPATIAL = 0.0001
    GAMMA_COLOR = 0.0001
    NUM_CLUSTER = 3
    INIT_CLUSTER = "k-means++" # "random" or "k-means++"

    # read image as numpy array
    image = np.asarray(Image.open(f"img/image{IMAGE}.png"))
    
    assert NUM_CLUSTER >= 2
    assert INIT_CLUSTER == "random" or INIT_CLUSTER == "k-means++"
    assert image.shape[0] == image.shape[1] == 100

    # k-means clustering
    kernel_kmeans(
        image,
        GAMMA_SPATIAL,
        GAMMA_COLOR,
        NUM_CLUSTER,
        INIT_CLUSTER,
        f"image{IMAGE}_{INIT_CLUSTER}_{NUM_CLUSTER}"
    )