import os
import numpy as np
from PIL import Image
import scipy.spatial.distance
import matplotlib.pyplot as plt

kernels = ['linear kernel', 'polynomial kernel', 'rbf kernel']

def visualize(title, target_data, target_name, eigenfaces, mean=None):
    
    # target_data shape = (10, 2500)
    # target_name shape = (10, )
    # eigenfaces shape = (2500, 25)
    # mean shape = (2500, )

    if title == "PCA":
        os.makedirs(name="PCA eigenfaces/eigenfaces", exist_ok=True)
        os.makedirs(name="PCA eigenfaces/reconstruction", exist_ok=True)
    else:
        os.makedirs(name="LDA fisherfaces/fisherfaces", exist_ok=True)
        os.makedirs(name="LDA fisherfaces/reconstruction", exist_ok=True)

    # eigenfaces
    if eigenfaces.shape[1] == 25:

        # save 25 eigenfaces in one image
        fig, axs = plt.subplots(5, 5)
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(eigenfaces[:, i*5 + j].reshape((50, 50)), cmap='gray')
                axs[i, j].axis('off')
        
        if title == "PCA":
            plt.savefig('PCA eigenfaces/eigenfaces/all.png')
        else:
            plt.savefig('LDA fisherfaces/fisherfaces/all.png')
    
        # save each eigenface as a single image
        # for i in range(25):
        #     plt.clf()
        #     plt.axis('off')
        #     plt.imshow(eigenfaces[:, i].reshape((50, 50)), cmap='gray')

        #     if title == "PCA":
        #         plt.savefig(f'PCA eigenfaces/eigenfaces/{i}.png')
        #     else:
        #         plt.savefig(f'LDA fisherfaces/fisherfaces/{i}.png')

    # reconstruction shape = (10, 2500)
    if mean is None:
        mean = np.zeros(target_data.shape[1])
    projection = (target_data - mean) @ eigenfaces
    reconstruction = projection @ eigenfaces.T + mean
    
    # reconstruction
    if reconstruction.shape[0] == 10:

        # save 10 reconstruction in one image
        fig, axs = plt.subplots(2, 5)
        for i in range(2):
            for j in range(5):
                axs[i, j].imshow(reconstruction[i*5 + j, :].reshape((50, 50)), cmap='gray')
                axs[i, j].axis('off')
        
        if title == "PCA":
            plt.savefig('PCA eigenfaces/reconstruction/all.png')
        else:
            plt.savefig('LDA fisherfaces/reconstruction/all.png')

        # save each reconstruction as a single image
        # for i in range(10):
        #     plt.clf()
        #     plt.axis('off')
        #     plt.imshow(reconstruction[i, :].reshape((50, 50)), cmap='gray')
            
        #     if title == "PCA":
        #         plt.savefig(f'PCA eigenfaces/reconstruction/{target_name[i]}.png')
        #     else:
        #         plt.savefig(f'LDA fisherfaces/reconstruction/{target_name[i]}.png')



def read_data(root_path, is_train):

    if is_train:
        root_path = os.path.join(root_path, "Training")
    else:
        root_path = os.path.join(root_path, "Testing")

    name = []
    data = []
    label = []

    for pgm_file in os.listdir(root_path):
        
        # image name
        name.append(pgm_file)

        # image data
        img = Image.open(os.path.join(root_path, pgm_file))
        img = img.resize((50, 50), Image.ANTIALIAS)
        img = np.array(img)
        data.append(img.ravel().astype(np.float64))

        # image label
        label.append(int(pgm_file[7:9]))

    return np.array(name), np.array(data), np.array(label)



def PCA(imgs, keep_nums):
    # imgs shape = (165, 2500)
    
    # mean of all images
    # shape = (2500, )
    mean = np.mean(imgs, axis=0)

    # extract distinguished features
    # shape = (165, 2500)
    imgs_feature = imgs - mean

    # covariance
    # shape = (165, 165)
    covariance = imgs_feature @ imgs_feature.T

    # eigen-decomposition
    # eigenvalue shape = (165, )
    # eigenvector shape = (165, 165)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)

    # calculate eigenfaces
    # shape = (2500, 165)
    eigenfaces = imgs_feature.T @ eigenvector

    # normalize eigenfaces
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
    
    # sort eigenfaces based on its corresponding eigenvalues
    idx = np.argsort(eigenvalue)[::-1]
    eigenfaces = eigenfaces[:, idx]

    # only keep first K eigenfaces
    eigenfaces = eigenfaces[:, :keep_nums].real
    return eigenfaces, mean



def LDA(imgs, labels, keep_nums):
    # imgs shape = (165, 2500)
    # labels shape = (165, )

    all_class = np.unique(labels)
    mean = np.mean(imgs, axis=0)

    # S_w: variance in class, shape = (2500, 2500)
    # S_b variance among classes, shape = (2500, 2500)
    S_w = np.zeros((imgs.shape[1], imgs.shape[1]), dtype=np.float64)
    S_b = np.zeros((imgs.shape[1], imgs.shape[1]), dtype=np.float64)

    # calculate S_w and S_b
    for c in all_class:
        imgs_subset = imgs[np.where(labels == c)[0], :]
        mean_subset = np.mean(imgs_subset, axis=0)
        S_w += (imgs_subset - mean_subset).T @ (imgs_subset - mean_subset)
        S_b += imgs_subset.shape[0] * ((mean_subset - mean).T @ (mean_subset - mean))
    
    # eigen-decomposition
    # eigenvalue shape = (2500, )
    # eigenvector shape = (2500, 2500)
    eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)

    # normalize
    for i in range(eigenvector.shape[1]):
        eigenvector[:, i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
    
    # sort eigenvectors based on its corresponding eigenvalues
    idx = np.argsort(eigenvalue)[::-1]
    fisherfaces = eigenvector[:, idx]

    # only keep first K eigenfaces
    fisherfaces = fisherfaces[:, :keep_nums].real
    return fisherfaces


def distance(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)

def face_recognition(train_projection, train_label, test_projection, test_label):

    # train_projection shape = (135, 25)
    # train_label shape = (135, )
    # test_projection shape = (30, 25)
    # test_label shape = (30, )

    test_distance = []
    for i in range(30):

        # calculate distance of a testing image between all training images
        dist_lst = []
        for j in range(135):
            dist = np.sum((test_projection[i] - train_projection[j]) ** 2)
            dist_lst.append([dist, train_label[j]])
        
        # sort data_lst based on distance
        dist_lst.sort(key=lambda x: x[0])
        test_distance.append(dist_lst)

    # k-nearest neighbor
    for k in range(1, 11):
        correct = 0
        for i in range(30):
            neighbors, count = np.unique(np.array([x[1] for x in test_distance[i][:k]]), return_counts=True)
            predict = neighbors[np.argmax(count)]
            if predict == test_label[i]:
                correct += 1
        print(f'[k={k}] {correct}/{30} => acc: {round(correct / 30, 2):.2f}')
    
    print()



def kernel_PCA(imgs, keep_nums, kernel_type):
    # imgs shape = (165, 2500)

    # compute kernel
    # kernel shape: (165, 165)
    if kernel_type == "linear":
        kernel = imgs @ imgs.T
    elif kernel_type == "rbf":
        kernel = np.exp(-1e-7 * scipy.spatial.distance.cdist(imgs, imgs, 'sqeuclidean'))
    else:
        raise Exception("Unknown kernel method")
    one_mat = np.ones((kernel.shape[0], kernel.shape[0]), dtype=np.float64) / kernel.shape[0]
    kernel = kernel - one_mat @ kernel - kernel @ one_mat + one_mat @ kernel @ one_mat
    
    # eigen-decomposition
    # eigenvalue shape = (165, )
    # eigenvector shape = (165, 165)
    eigenvalue, eigenvector = np.linalg.eigh(kernel)

    # normalize
    for i in range(eigenvector.shape[1]):
        eigenvector[:, i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
    
    # sort eigenfaces based on its corresponding eigenvalues
    idx = np.argsort(eigenvalue)[::-1]
    eigenvector = eigenvector[:, idx]

    # only keep first K eigenfaces
    eigenvector = eigenvector[:, :keep_nums].real
    return kernel @ eigenvector



def kernel_LDA(imgs, labels, keep_nums, kernel_type):
    # imgs shape = (165, 2500)
    # labels shape = (165, )

    # compute kernel
    # kernel shape: (165, 165)
    if kernel_type == "linear":
        kernel = imgs @ imgs.T
    elif kernel_type == "rbf":
        kernel = np.exp(-1e-7 * scipy.spatial.distance.cdist(imgs, imgs, 'sqeuclidean'))
    else:
        raise Exception("Unknown kernel method.")

    all_class = np.unique(labels)
    mean = np.mean(kernel, axis=0)
    S_w = np.zeros((kernel.shape[0], kernel.shape[0]), dtype=np.float64)
    S_b = np.zeros((kernel.shape[0], kernel.shape[0]), dtype=np.float64)
    
    # calculate S_w and S_b
    for c in all_class:
        imgs_subset = kernel[np.where(labels == c)[0], :]
        mean_subset = np.mean(imgs_subset, axis=0)
        S_w += imgs_subset.T @ (np.eye(imgs_subset.shape[0]) - (np.ones((imgs_subset.shape[0], imgs_subset.shape[0]), dtype=np.float64) / imgs_subset.shape[0])) @ imgs_subset
        S_b += imgs_subset.shape[0] * ((mean_subset - mean).T @ (mean_subset - mean))
    
    # eigen-decomposition
    # eigenvalue shape = (165, )
    # eigenvector shape = (165, 165)
    eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)

    # normalize
    for i in range(eigenvector.shape[1]):
        eigenvector[:, i] = eigenvector[:, i] / np.linalg.norm(eigenvector[:, i])
    
    # sort eigenvectors based on its corresponding eigenvalues
    idx = np.argsort(eigenvalue)[::-1]
    fisherfaces = eigenvector[:, idx]

    # only keep first K eigenfaces
    fisherfaces = fisherfaces[:, :keep_nums].real
    return kernel @ fisherfaces



if __name__ == '__main__':

    '''
    TASK 1: Show PCA Eigenfaces, LDA Fisherfaces and Reconstruction
    TASK 2: Face Recognition with PCA and LDA
    TASK 3: Face Recognition with Kernel PCA and Kernel LDA
    '''
    TASK = 3
    
    train_name, train_data, train_label = read_data(root_path='Yale_Face_Database', is_train=True)
    test_name, test_data, test_label = read_data(root_path='Yale_Face_Database', is_train=False)

    # train_data shape = (135, 2500)
    # test_data shape = (30, 2500)
    # all_data shape = (165, 2500)
    all_data = np.vstack((train_data, test_data))

    # train_name shape = (135, )
    # test_name shape = (30, )
    # all_name shape = (165, )
    all_name = np.hstack((train_name, test_name))

    # train_label shape = (135, )
    # test_label shape = (30, )
    # all_label shape = (165, )
    all_label = np.hstack((train_label, test_label))

    if TASK == 1:
        # randomly select 10 images
        target_idx = np.random.choice(all_data.shape[0], 10)
        target_data = all_data[target_idx]
        target_name = all_name[target_idx]

        # PCA eigenfaces: https://laid.delanover.com/explanation-face-recognition-using-eigenfaces/
        print("PCA eigenfaces")
        # eigenfaces shape = (2500, 25)
        # mean shape = (2500, )
        eigenfaces, mean = PCA(all_data, 25)
        visualize("PCA", target_data, target_name, eigenfaces, mean)

        # LDA fisherfaces
        print("LDA fisherfaces")
        # fisherfaces shape = (2500, 25)
        fisherfaces = LDA(all_data, all_label, 25)
        visualize("LDA", target_data, target_name, fisherfaces)

    elif TASK == 2:
        print("Face Recognition: PCA")
        eigenfaces, mean = PCA(all_data, 25)
        train_projection = (train_data - mean) @ eigenfaces
        test_projetion = (test_data - mean) @ eigenfaces
        face_recognition(train_projection, train_label, test_projetion, test_label)

        print("Face Recognition: LDA")
        fisherfaces = LDA(all_data, all_label, 25)
        train_projection = train_data @ fisherfaces
        test_projetion = test_data @ fisherfaces
        face_recognition(train_projection, train_label, test_projetion, test_label)

    elif TASK == 3:
        kernel_type = "rbf"
        
        print(f"Face Recognition: Kernel PCA ({kernel_type})")
        kernel_coord = kernel_PCA(all_data, 25, kernel_type)
        train_coord = kernel_coord[:135, :]
        test_coord = kernel_coord[135:, :]
        face_recognition(train_coord, train_label, test_coord, test_label)

        print(f"Face Recognition: Kernel LDA ({kernel_type})")
        kernel_coord = kernel_LDA(all_data, all_label, 25, kernel_type)
        train_coord = kernel_coord[:135]
        test_coord = kernel_coord[135:]
        face_recognition(train_coord, train_label, test_coord, test_label)