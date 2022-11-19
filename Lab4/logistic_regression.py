import numpy as np

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

def confusion_matrix(X_train, y_train, weight_gradient):
    result = np.zeros((2, 2))
    for idx, row in enumerate(X_train):
        pred = np.matmul(row.reshape(1, -1), weight_gradient)[0][0]
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
        

def calc_gradient(X_train, y_train, weight):
    deno = 1 + np.exp(-np.matmul(X_train, weight))
    return np.matmul(np.transpose(X_train), (y_train - (1 / deno)))

def gradient_descent(X_train, y_train):

    # initialize weight
    current_weight = np.random.rand(3, 1)
    iter_limit = 10000
    iter_count = 0

    while True:
        grad = calc_gradient(X_train, y_train, current_weight)
        new_weight = current_weight + grad

        # if weight does not change a lot, it converges
        diff = np.linalg.norm(new_weight - current_weight)
        if diff < 0.001 or iter_count >= iter_limit:
            break

        current_weight = new_weight
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
    result = confusion_matrix(X_train, y_train, weight_gradient)

    print(weight_gradient)
    print(result)