import numpy as np
from PIL import Image
import os, re
import scipy.spatial.distance
from datetime import datetime
import matplotlib.pyplot as plt

# SHAPE = (195, 231)
# SHAPE = (60, 60)
SHAPE = (50, 50)

K = [1, 3, 5, 7, 9, 11]
kernels = ['linear kernel', 'polynomial kernel', 'rbf kernel']

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

def LDA(X, label, dims):
    (n, d) = X.shape
    label = np.asarray(label)
    c = np.unique(label)
    mu = np.mean(X, axis=0)
    S_w = np.zeros((d, d), dtype=np.float64)
    S_b = np.zeros((d, d), dtype=np.float64)
    for i in c:
        X_i = X[np.where(label == i)[0], :]
        mu_i = np.mean(X_i, axis=0)
        S_w += (X_i - mu_i).T @ (X_i - mu_i)
        S_b += X_i.shape[0] * ((mu_i - mu).T @ (mu_i - mu))
    eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dims].real
    return W

# def PCALDA(X, label, dims):
#     label = np.asarray(label)
#     (n, d) = X.shape
#     c = len(np.unique(label))
#     [eigen_vec_pca, mu_pca] = PCA(X, n - c)
#     projection = (X - mu_pca) @ eigen_vec_pca
#     eigen_vec_lda = LDA(projection, label, dims)
#     eigen_vec = eigen_vec_pca @ eigen_vec_lda
#     return [eigen_vec, mu_pca]

def linearKernel(X):
    return X @ X.T

def polynomialKernel(X, gamma, coef, degree):
    return np.power(gamma * (X @ X.T) + coef, degree)

def rbfKernel(X, gamma):
    return np.exp(-gamma * scipy.spatial.distance.cdist(X, X, 'sqeuclidean'))

def getKernel(X, kernel_type):
    if kernel_type == 1:
        kernel = linearKernel(X)
    elif kernel_type == 2:
        kernel = polynomialKernel(X, 5, 10, 2)
    else:
        kernel = rbfKernel(X, 1e-7)
    return kernel

def kernelPCA(X, dims, kernel_type):
    kernel = getKernel(X, kernel_type)
    n = kernel.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    kernel = kernel - one @ kernel - kernel @ one + one @ kernel @ one
    eigen_val, eigen_vec = np.linalg.eigh(kernel)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dims].real
    return kernel @ W

# def kernelLDA(X, label, dims, kernel_type):
#     label = np.asarray(label)
#     c = np.unique(label)
#     kernel = getKernel(X, kernel_type)
#     n = kernel.shape[0]
#     mu = np.mean(kernel, axis=0)
#     N = np.zeros((n, n), dtype=np.float64)
#     M = np.zeros((n, n), dtype=np.float64)
#     for i in c:
#         K_i = kernel[np.where(label == i)[0], :]
#         l = K_i.shape[0]
#         mu_i = np.mean(K_i, axis=0)
#         N += K_i.T @ (np.eye(l) - (np.ones((l, l), dtype=np.float64) / l)) @ K_i
#         M += l * ((mu_i - mu).T @ (mu_i - mu))
#     eigen_val, eigen_vec = np.linalg.eig(np.linalg.pinv(N) @ M)
#     for i in range(eigen_vec.shape[1]):
#         eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
#     idx = np.argsort(eigen_val)[::-1]
#     W = eigen_vec[:, idx][:, :dims].real
#     return kernel @ W

def draw(target_data, target_filename, title, W, mu=None):
    if mu is None:
        mu = np.zeros(target_data.shape[1])
    projection = (target_data - mu) @ W
    reconstruction = projection @ W.T + mu
    folder = f"{title}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(folder)
    os.mkdir(f'{folder}/{title}')
    if W.shape[1] == 25:
        plt.clf()
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                plt.subplot(5, 5, idx + 1)
                plt.imshow(W[:, idx].reshape(SHAPE[::-1]), cmap='gray')
                plt.axis('off')
        plt.savefig(f'./{folder}/{title}/{title}.png')
    for i in range(W.shape[1]):
        plt.clf()
        plt.title(f'{title}_{i + 1}')
        plt.imshow(W[:, i].reshape(SHAPE[::-1]), cmap='gray')
        plt.savefig(f'./{folder}/{title}/{title}_{i + 1}.png')
    
    if reconstruction.shape[0] == 10:
        plt.clf()
        for i in range(2):
            for j in range(5):
                idx = i * 5 + j
                plt.subplot(2, 5, idx + 1)
                plt.imshow(reconstruction[idx].reshape(SHAPE[::-1]), cmap='gray')
                plt.axis('off')
        plt.savefig(f'./{folder}/reconstruction.png')
    for i in range(reconstruction.shape[0]):
        plt.clf()
        plt.title(target_filename[i])
        plt.imshow(reconstruction[i].reshape(SHAPE[::-1]), cmap='gray')
        plt.savefig(f'./{folder}/{target_filename[i]}.png')

def distance(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)

def faceRecognition(X, X_label, test, test_label, method, kernel_type=None):
    if kernel_type is None:
        print(f'Face recognition with {method} and KNN:')
    else:
        print(f'Face recognition with Kernel {method}({kernels[kernel_type - 1]}) and KNN:')
    dist_mat = []
    for i in range(test.shape[0]):
        dist = []
        for j in range(X.shape[0]):
            dist.append((distance(X[j], test[i]), X_label[j]))
        dist.sort(key=lambda x: x[0])
        dist_mat.append(dist)
    for k in K:
        correct = 0
        total = test.shape[0]
        for i in range(test.shape[0]):
            dist = dist_mat[i]
            neighbor = np.asarray([x[1] for x in dist[:k]])
            neighbor, count = np.unique(neighbor, return_counts=True)
            predict = neighbor[np.argmax(count)]
            if predict == test_label[i]:
                correct += 1
        print(f'K={k:>2}, accuracy: {correct / total:>.3f} ({correct}/{total})')
    print()

if __name__ == '__main__':

    '''
    TASK 1: Show PCA Eigenfaces, LDA Fisherfaces and Reconstruction
    '''
    TASK = 1
    
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
        target_filename = all_name[target_idx]

        # PCA eigenfaces: https://laid.delanover.com/explanation-face-recognition-using-eigenfaces/
        eigenfaces, mean = PCA(all_data, 25)
        draw(target_data, target_filename, 'pca_eigenface', eigenfaces, mean)

        exit()
        # LDA fisherfaces
        W = LDA(data, label, 25)
        draw(target_data, target_filename, 'lda_fisherface', W)

        # elif TASK == '2':
        #     W, mu = PCA(data, 25)
        #     X_proj = (X - mu) @ W
        #     test_proj = (test - mu) @ W
        #     faceRecognition(X_proj, X_label, test_proj, test_label, 'PCA')

        #     W = LDA(data, label, 25)
        #     X_proj = X @ W
        #     test_proj = test @ W
        #     faceRecognition(X_proj, X_label, test_proj, test_label, 'LDA')

        # elif TASK == '3':
        #     kernel_type = int(input('1) Linear kernel\n2) Polynomial kernel\n3) RBF kernel\nChoose a kernel: '))
            
        #     new_coor = kernelPCA(data, 25, kernel_type)
        #     new_X = new_coor[:X.shape[0], :]
        #     new_test = new_coor[X.shape[0]:, :]
        #     faceRecognition(new_X, X_label, new_test, test_label, 'PCA', kernel_type)

        #     print('KernelLDA not implemented')
            # new_coor = kernelLDA(data, label, 25, kernel_type)
            # new_X = new_coor[:X.shape[0]]
            # new_test = new_coor[X.shape[0]:]
            # faceRecognition(new_X, X_label, new_test, test_label, 'LDA', kernel_type)