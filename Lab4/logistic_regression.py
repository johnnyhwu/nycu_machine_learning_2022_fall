import numpy as np
from scipy.special import expit, expm1
from scipy.linalg import inv
import matplotlib.pyplot as plt

def univariate_gaussian(mean, variance):
    points = np.random.uniform(0.0, 1.0, 12)
    total_sum = np.sum(points)
    standard_normal = total_sum - 6
    return standard_normal * np.sqrt(variance) + mean

def make_dataset(x1, y1, x2, y2, n):
    class_0_feat, class_1_feat = [], []
    for _ in range(n):
        x_val = univariate_gaussian(x1[0], x1[1])
        y_val = univariate_gaussian(y1[0], y1[1])
        class_0_feat.append([x_val, y_val])

        x_val = univariate_gaussian(x2[0], x2[1])
        y_val = univariate_gaussian(y2[0], y2[1])
        class_1_feat.append([x_val, y_val])
    
    class_0_feat = np.array(class_0_feat)
    class_0_trg = np.zeros((n, 1))

    class_1_feat = np.array(class_1_feat)
    class_1_trg = np.ones((n, 1))
    
    feat = np.concatenate([class_0_feat, class_1_feat], axis=0)
    trg = np.concatenate([class_0_trg, class_1_trg], axis=0)

    return feat, trg

def show_result(X_train, y_train, weight, title):
    matrix = confusion_matrix(X_train, y_train, weight)

    print(title)

    print("\nweight:")
    print(weight)

    print("\nconfusion matrix: ")
    print("\t\tPredict cluster 1 Predict cluster 2")
    print(f"Is cluster 1\t\t{matrix[0][0]}\t\t{matrix[0][1]}")
    print(f"Is cluster 2\t\t{matrix[1][0]}\t\t{matrix[1][1]}")

    print(f"\nSensitivity (Successfully predict cluster 1): {(matrix[0][0] / (matrix[0][0] + matrix[0][1])):.4f}")
    print(f"Specificity (Successfully predict cluster 2): {(matrix[1][1] / (matrix[1][0] + matrix[1][1])):.4f}")


def confusion_matrix(X_train, y_train, weight):
    result = np.zeros((2, 2))
    for idx, row in enumerate(X_train):
        pred = np.matmul(row.reshape(1, -1), weight)[0][0]
        if pred < 0:
            if y_train[idx][0] == 0:
                result[0][0] += 1
            else:
                result[1][0] += 1
        else:
            if y_train[idx][0] == 0:
                result[0][1] += 1
            else:
                result[1][1] += 1
    return result

def vis_result(X_train, y_train, weight_gradient, weight_newton):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    num_sample = int(X_train.shape[0] / 2)

    # groundtruth
    axes[0].set_title("Groundtruth")
    axes[0].scatter(x=X_train[0:num_sample, 0], y=X_train[0:num_sample, 1], c="red")
    axes[0].scatter(x=X_train[num_sample:, 0], y=X_train[num_sample:, 1], c="blue")

    # gradient descent
    pred_0_idxs, pred_1_idxs = [], []
    for idx, row in enumerate(X_train):
        pred = np.matmul(row.reshape(1, -1), weight_gradient)
        if pred < 0:
            pred_0_idxs.append(idx)
        else:
            pred_1_idxs.append(idx)
    axes[1].set_title("Gradient Descent")
    axes[1].scatter(x=X_train[pred_0_idxs, 0], y=X_train[pred_0_idxs, 1], c="red")
    axes[1].scatter(x=X_train[pred_1_idxs, 0], y=X_train[pred_1_idxs, 1], c="blue")

    # newton's method
    pred_0_idxs, pred_1_idxs = [], []
    for idx, row in enumerate(X_train):
        pred = np.matmul(row.reshape(1, -1), weight_newton)
        if pred < 0:
            pred_0_idxs.append(idx)
        else:
            pred_1_idxs.append(idx)
    axes[2].set_title("Newton Method")
    axes[2].scatter(x=X_train[pred_0_idxs, 0], y=X_train[pred_0_idxs, 1], c="red")
    axes[2].scatter(x=X_train[pred_1_idxs, 0], y=X_train[pred_1_idxs, 1], c="blue")

    plt.show()
        

def calc_gradient(X_train, y_train, weight):
    return np.matmul(np.transpose(X_train), (y_train - expit(np.matmul(X_train, weight))))

def gradient_descent(X_train, y_train):

    # initialize weight
    current_weight = np.zeros((3, 1))
    iter_limit = 10000
    iter_count = 0

    while True:
        # calculate gradient
        grad = calc_gradient(X_train, y_train, current_weight)

        # update weight
        new_weight = current_weight + grad

        # check if converges
        diff = np.linalg.norm(new_weight - current_weight)
        if diff < 0.001 or iter_count >= iter_limit:
            break

        current_weight = new_weight.copy()
        iter_count += 1
    
    return current_weight

def calc_hessian(X_train, weight):
    d = np.zeros((X_train.shape[0], X_train.shape[0]))
    product = np.matmul(X_train, weight)
    diagonal = (expm1(-product) + 1) * np.power(expit(product), 2)
    np.fill_diagonal(d, diagonal)
    hessian = np.matmul(np.transpose(X_train), (np.matmul(d, X_train)))
    return hessian

def newton_method(X_train, y_train):

    # initialize weight
    current_weight = np.zeros((3, 1))
    iter_limit = 10000
    iter_count = 0

    while True:
        hessian = calc_hessian(X_train, current_weight)

        try:
            hessian_inv = inv(hessian)
            grad = calc_gradient(X_train, y_train, current_weight)
            new_weight = current_weight + np.matmul(hessian_inv, grad)
        except:
            grad = calc_gradient(X_train, y_train, current_weight)
            new_weight = current_weight + grad

        # check if converges
        diff = np.linalg.norm(new_weight - current_weight)
        if diff < 0.001 or iter_count >= iter_limit:
            break

        current_weight = new_weight.copy()
        iter_count += 1

    return current_weight


def test_case(id):
    if id == 1:
        num_sample = 50
        x1 = (1, 2)
        y1 = (1, 2)
        x2 = (10, 2)
        y2 = (10, 2)
    elif id == 2:
        num_sample = 50
        x1 = (1, 2)
        y1 = (1, 2)
        x2 = (3, 4)
        y2 = (3, 4)
    
    return num_sample, x1, y1, x2, y2

if __name__ == "__main__":

    USER_INPUT = False
    SEED = 47

    # reproducible output
    np.random.seed(SEED)

    # user input
    if(USER_INPUT):
        num_sample = int(input("number of samples: "))
        mean_x1 = float(input("mean of x1: "))
        variance_x1 = float(input("variance of x1: "))
        mean_y1 = float(input("mean of y1: "))
        variance_y1 = float(input("variance of y1: "))
        mean_x2 = float(input("mean of x2: "))
        variance_x2 = float(input("variance of x2: "))
        mean_y2 = float(input("mean of y2: "))
        variance_y2 = float(input("variance of y2: "))

        x1=(mean_x1, variance_x1)
        y1=(mean_y1, variance_y1)
        x2=(mean_x2, variance_x2)
        y2=(mean_y2, variance_y2)
    else:
        id = int(input("test case id: "))
        num_sample, x1, y1, x2, y2 = test_case(id)

    # make dataset
    X_train, y_train = make_dataset(x1, y1, x2, y2, num_sample)

    # create design matrix
    X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    
    # gradient descent
    weight_gradient = gradient_descent(X_train, y_train)
    show_result(X_train, y_train, weight_gradient, "Gradient Descent")

    print("\n-----------------------------------------------------\n")

    # newton's method
    weight_newton = newton_method(X_train, y_train)
    show_result(X_train, y_train, weight_newton, "Newton's Method")

    # visialize result
    vis_result(X_train, y_train, weight_gradient, weight_newton)
