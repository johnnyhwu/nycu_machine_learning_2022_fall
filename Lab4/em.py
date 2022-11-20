import argparse
import numpy as np
import numba as nb
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True)

# py EM_algorithm.py

SEED = 42
np.random.seed(SEED)

def load_dataset():
    img_path = "train-images-idx3-ubyte"
    label_path = "train-labels-idx1-ubyte"
    
    # read image meta data and image pixel value
    with open(img_path, "rb") as f:
        _, num_img, num_row, num_col = np.fromfile(file=f, dtype=">i4", count=4)
        all_imgs = np.fromfile(file=f, dtype=">B", count=-1)
    
    # read image label
    with open(label_path, "rb") as f:
        _ = np.fromfile(file=f, dtype=">i4", count=2)
        all_labels = np.fromfile(file=f, dtype=">B", count=-1)
    
    # reshape image to Nx28x28
    all_imgs = np.reshape(all_imgs, (num_img, num_row, num_col))

    return (all_imgs, all_labels)

@nb.jit
def expectation_step(all_imgs, cluster_pixel_prob, cluster_prob, img_cluster_prob):

    """
    計算每一張圖片屬於每一個 Cluster
    """

    for img_idx in range(n_img):

        # 先將這張圖片的屬於每一個 Cluster 的機率初始化為 1
        img_cluster_prob[img_idx, :] = 1

        # 計算屬於每一個 Cluster 的機率
        for cluster_idx in range(n_cluster):

            # 利用這張圖片的每一個 Featue (Pixel) 來估計
            for pixel_idx in range(n_pixel):

                if all_imgs[img_idx][pixel_idx] == 1:
                    img_cluster_prob[img_idx, cluster_idx] *= cluster_pixel_prob[cluster_idx][pixel_idx]
                else:
                    img_cluster_prob[img_idx, cluster_idx] *= (1 - cluster_pixel_prob[cluster_idx][pixel_idx])
            
            # 還要再乘以每一個 Cluster 出現的機率
            img_cluster_prob[img_idx, cluster_idx] *= cluster_prob[cluster_idx][0]
        
        # 讓這張圖片出現在每一個 Cluster 之機率總和為 1
        marginal_sum = img_cluster_prob[img_idx, :].sum()
        if marginal_sum == 0:
            marginal_sum = 1
        img_cluster_prob[img_idx, :] /= marginal_sum

    return img_cluster_prob

def maximization_step(all_imgs, cluster_pixel_prob, cluster_prob, img_cluster_prob):

    """
    利用每張圖片出現在每一個 Cluster 的機率，來重新估計：
    1. 每一個 Cluster 出現的機率
    2. 每一個 Cluster 中針對圖片每一個 Feature 的機率
    """
    
    N_cluster = np.sum(img_cluster_prob, axis=0)

    # 處理每一個 Cluster
    for cluster_idx in range(n_cluster):

        # 計算這個 Cluster 出現的機率
        cluster_prob[cluster_idx][0] = N_cluster[cluster_idx] / n_img
        if cluster_prob[cluster_idx][0] == 0:
            cluster_prob[cluster_idx][0] = 1

        # 計算 Clsuter 中針對每一個 Feature 的機率
        for pixel_idx in range(n_pixel):
            sum_pixel = np.dot(all_imgs[:, pixel_idx], img_cluster_prob[:, cluster_idx])
            marginal = N_cluster[cluster_idx]
            if marginal == 0:
                marginal = 1
            cluster_pixel_prob[cluster_idx][pixel_idx] = sum_pixel / marginal

    return cluster_pixel_prob, cluster_prob

def imagine_digit(cluster_pixel_prob):
    digits = cluster_pixel_prob.copy()
    digits = digits.reshape((n_cluster, 28, 28))

    for cluster_idx in range(n_cluster):
        print(f"\nClass: {cluster_idx}")

        digit = digits[cluster_idx, :, :]
        digit[digit < 0.5] = 0
        digit[digit >= 0.5] = 1
        digit = digit.astype("int")

        for r in range(28):
            for c in range(28):
                print(digit[r, c], end=" ")
            print()

@nb.jit
# def label_cluster(train_bin, train_label, mu, pi):
def label_cluster(all_imgs, all_labels, cluster_pixel_prob, cluster_prob):

    """
    判斷每一個 Cluster 屬於 0 ~ 9 哪一個 Label 時，
    會先建立一個 Table 統計每一個 Cluster 中出現每一個 Label 的次數，
    最後，每次從 Table 中定位到最大的數字，來決定每一個 Cluster 的 Label
    """

    table = np.zeros(shape=(n_cluster, n_cluster), dtype=np.int)

    for img_idx in range(n_img):

        # 計算此張圖片屬於哪一個 Cluster
        num_sum = np.full((n_cluster), 1, dtype=np.float64)
        for num_idx in range(n_cluster):
            for pixel_idx in range(n_pixel):
                if all_imgs[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= cluster_pixel_prob[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - cluster_pixel_prob[num_idx][pixel_idx])
            num_sum[num_idx] *= cluster_prob[num_idx][0]

        # 紀錄每一個 Cluster 中，每一個 Label 出現的次數
        table[all_labels[img_idx][0]][np.argmax(num_sum)] += 1
    
    print(table)

    label2cluster = np.full((n_cluster), -1, dtype=np.int)
    # label2cluster: [1 5 2 9 8 6 0 7 3 4]

    for num_idx in range(n_cluster):
        label_idx, cluster_idx = np.unravel_index(np.argmax(table, axis=None), table.shape)
        label2cluster[label_idx] = cluster_idx
        for i in range(n_cluster):
            table[i][cluster_idx] = -1
            table[label_idx][i] = -1
    
    print(label2cluster)

    return label2cluster

# def print_label(relation, mu):
def print_label(label2cluster, cluster_pixel_prob):
    for class_idx in range(n_cluster):

        cluster = label2cluster[class_idx]
        print("\nLabeled class : ", class_idx)

        digit = cluster_pixel_prob[cluster, :].copy()
        digit = digit.reshape((28, 28))
        digit[digit < 0.5] = 0
        digit[digit >= 0.5] = 1
        digit = digit.astype("int")

        for r in range(28):
            for c in range(28):
                print(digit[r, c], end=" ")
            print()

@nb.jit
def print_confusion_matrix(
        all_imgs, 
        all_labels, 
        cluster_pixel_prob, 
        cluster_prob, 
        label2cluster
    ):
    
    error = n_img
    confusion_matrix = np.zeros(shape=(n_cluster, 4), dtype=np.int)

    for img_idx in range(n_img):

        num_sum = np.full((n_cluster), 1, dtype=np.float64)

        for num_idx in range(n_cluster):
            for pixel_idx in range(n_pixel):
                if all_imgs[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= cluster_pixel_prob[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - cluster_pixel_prob[num_idx][pixel_idx])
            num_sum[num_idx] *= cluster_prob[num_idx][0]

        predict_cluster = np.argmax(num_sum)
        predict_label = np.where(label2cluster == predict_cluster)

        for num_idx in range(n_cluster):
            if num_idx == all_labels[img_idx][0]:
                if num_idx == predict_label[0]:
                    error -= 1
                    confusion_matrix[num_idx][0] += 1
                else:
                    confusion_matrix[num_idx][3] += 1
            else:
                if num_idx == predict_label[0]:
                    confusion_matrix[num_idx][1] += 1
                else:
                    confusion_matrix[num_idx][2] += 1

    for num_idx in range(n_cluster):
        print("Confusion matrix {}:".format(num_idx))
        print("\t\tPredict number {}\tPredict not number {}".format(num_idx, num_idx))
        print("Is number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][0], confusion_matrix[num_idx][3]))
        print("Isn't number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][1], confusion_matrix[num_idx][2]))
        print("Sensitivity (Successfully predict number {}): {}".format(num_idx, confusion_matrix[num_idx][0] / (confusion_matrix[num_idx][0] + confusion_matrix[num_idx][3])))
        print("Specificity (Successfully predict not number {}): {}".format(num_idx, confusion_matrix[num_idx][2] / (confusion_matrix[num_idx][2] + confusion_matrix[num_idx][1])))
        print("---------------------------------------------------------------\n")

    return error



if __name__ == '__main__':
    (all_imgs, all_labels) = load_dataset()
    all_imgs[all_imgs < 128] = 0
    all_imgs[all_imgs >= 128] = 1
    all_imgs = all_imgs.astype("int")
    all_imgs = all_imgs.reshape((60000, 28*28))
    all_labels = all_labels.reshape(60000, 1)

    global n_cluster, n_pixel, n_img
    n_cluster = 10
    n_pixel = 28*28
    n_img = 60000

    cluster_prob = np.random.random_sample((n_cluster, 1))
    cluster_pixel_prob = np.random.random_sample((n_cluster, n_pixel))
    cluster_pixel_prob_prev = cluster_pixel_prob.copy()
    img_cluster_prob = np.random.random_sample((n_img, n_cluster))

    iter_count = 0
    while True:
        iter_count += 1

        img_cluster_prob = expectation_step(all_imgs, cluster_pixel_prob, cluster_prob, img_cluster_prob)
        cluster_pixel_prob, cluster_prob = maximization_step(all_imgs, cluster_pixel_prob, cluster_prob, img_cluster_prob)
        diff = np.abs(cluster_pixel_prob - cluster_pixel_prob_prev).sum()

        imagine_digit(cluster_pixel_prob)
        print(f"\nIteration: {iter_count}, Difference: {diff}")
        print("---------------------------------------------------------------\n\n")

        if diff < 10 or iter_count == 10:
            break

        cluster_pixel_prob_prev = cluster_pixel_prob.copy()


    label2cluster = label_cluster(all_imgs, all_labels, cluster_pixel_prob, cluster_prob)
    print_label(label2cluster, cluster_pixel_prob)

    print("---------------------------------------------------------------\n")

    error = print_confusion_matrix(
        all_imgs, 
        all_labels, 
        cluster_pixel_prob, 
        cluster_prob, 
        label2cluster
    )
    print("Total iteration to converge: {}".format(iter_count))
    print("Total error rate: {}".format(error / n_img))