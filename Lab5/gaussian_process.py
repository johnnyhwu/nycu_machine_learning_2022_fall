import scipy
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(path):
    X_train, y_train = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            x, y = line.split(' ')
            X_train.append(float(x))
            y_train.append(float(y))
    X_train = np.array(X_train, dtype=np.float64).reshape(-1, 1)
    y_train = np.array(y_train, dtype=np.float64).reshape(-1, 1)
    return X_train, y_train


def rational_quadratic_kernel(X1, X2, sigma, alpha, length_scale):
    nume = np.power(X1, 2) + np.power(X2, 2).flatten() - 2 * np.matmul(X1, np.transpose(X2))
    return (sigma ** 2) * ((1 + nume / (2 * alpha * (length_scale ** 2))) ** (-1 * alpha))
    

def negative_log_likelihood(params, X_train, y_train, beta):
    sigma, alpha, length_scale = params.flatten()

    kernel_Xtr_Xtr = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale)
    C = kernel_Xtr_Xtr + (1 / beta) * np.identity(X_train.shape[0], dtype=np.float64)

    nll = 0.5 * np.log(np.linalg.det(C))
    nll += 0.5 * np.matmul(np.matmul(np.transpose(y_train), np.linalg.inv(C)), y_train)
    nll += 0.5 * X_train.shape[0] * np.log(2 * np.pi)
    return nll

def gaussian_process(
        X_train,
        y_train,
        X_test,
        beta,
        sigma,
        alpha,
        length_scale
    ):

    kernel_Xtr_Xtr = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale)
    kernel_Xtr_Xte = rational_quadratic_kernel(X_train, X_test, sigma, alpha, length_scale)
    kernel_Xte_Xte = rational_quadratic_kernel(X_test, X_test, sigma, alpha, length_scale) 
    C = kernel_Xtr_Xtr + (1 / beta) * np.identity(X_train.shape[0], dtype=np.float64)

    mean_Xte = np.matmul(np.matmul(np.transpose(kernel_Xtr_Xte), np.linalg.inv(C)), y_train)
    variance_Xte = kernel_Xte_Xte + (1 / beta) * np.identity(len(X_test), dtype=np.float64)
    variance_Xte -= np.matmul(np.matmul(np.transpose(kernel_Xtr_Xte), np.linalg.inv(C)), kernel_Xtr_Xte)
    
    plt.plot(X_test, mean_Xte, color='red')
    plt.scatter(X_train, y_train, color='blue', s=5)
    
    interval = 1.96 * np.sqrt(np.diag(variance_Xte))
    X_test = X_test.flatten()
    mean_Xte = mean_Xte.flatten()

    plt.plot(X_test, mean_Xte + interval, color='coral')
    plt.plot(X_test, mean_Xte - interval, color='coral')
    plt.fill_between(X_test, mean_Xte + interval, mean_Xte - interval, color='coral', alpha=0.1)

    plt.title(f'sigma: {sigma:.5f}, alpha: {alpha:.5f}, length scale: {length_scale:.5f}')
    plt.xlim(-60, 60)
    plt.show()

if __name__ == '__main__':

    # load training dataset: shape of X_train and y_train is (34, 1)
    X_train, y_train = load_dataset('data/input.data')

    # generate testing samples between -60 and 60: shape of X_test is (1000, 1)
    X_test = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)

    # default parameter
    beta = 5
    sigma = 1
    alpha = 1
    length_scale = 1

    # apply gaussian process
    gaussian_process(
        X_train,
        y_train,
        X_test,
        beta,
        sigma,
        alpha,
        length_scale
    )
    
    opt = scipy.optimize.minimize(
        negative_log_likelihood, 
        [sigma, alpha, length_scale], 
        bounds=((1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6)), 
        args=(X_train, y_train, beta)
    )

    best_sigma = opt.x[0]
    best_alpha = opt.x[1]
    best_length_scale = opt.x[2]
    gaussian_process(
        X_train,
        y_train,
        X_test,
        beta,
        best_sigma,
        best_alpha,
        best_length_scale
    )