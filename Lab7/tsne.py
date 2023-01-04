import sys, os
import numpy as np
import seaborn as sns
import scipy.spatial.distance
import matplotlib.pyplot as plt

def Hbeta(D, beta):
    
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X, tol, perplexity):
    
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print('Computing pairwise distances...')
    (n, d) = X.shape
    D = scipy.spatial.distance.cdist(X, X, 'sqeuclidean')
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))
        
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
    # Return final P-matrix
    print(f'Mean value of sigma: {np.mean(np.sqrt(1 / beta))}')
    return P

def pca(X, dims):

    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print('Preprocessing the data using PCA...')
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, axis=0), (n, 1))
    (eigen_val, eigen_vec) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, eigen_vec[:, 0:dims])
    return Y

def SNE(data, label, target_dim, original_dim, perplexity, method):
    
    # data shape = (2500, 784)
    # label = (784, )

    # dimension reduction on data ith PCA
    # data shape = (2500, 50)
    data = pca(data, original_dim).real
    num_sample = data.shape[0]

    # Initialize variables
    max_iter = 500
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(num_sample, target_dim)
    dY = np.zeros((num_sample, target_dim))
    iY = np.zeros((num_sample, target_dim))
    gains = np.ones((num_sample, target_dim))
    

    # Compute P-values
    P = x2p(data, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4                   # early exaggeration
    P = np.maximum(P, 1e-12)

    for itr in range(max_iter):

        # Compute pairwise affinities
        if method == 'T-SNE':
            num = 1 / (1 + scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
        else:
            num = np.exp(-1 * scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
        num[range(num_sample), range(num_sample)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(num_sample):
            if method == 'T-SNE':
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (target_dim, 1)).T * (Y[i, :] - Y), axis=0)
            else:
                dY[i, :] = np.sum(np.tile(PQ[:, i], (target_dim, 1)).T * (Y[i, :] - Y), axis=0)

        # Perform the update
        if itr < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (num_sample, 1))

        if itr % 10 == 0:
            visualize(Y, label, itr, method, perplexity)

        # Compute current value of cost function
        if (itr + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f'Iteration {itr + 1}: error is {C}')

        # Stop lying about P-values
        if itr == 100:
            P = P / 4

    return Y, P, Q

def visualize(Y, labels, itr, method, perplexity):
    plt.clf()
    scatter = plt.scatter(Y[:, 0], Y[:, 1], 10, labels)
    plt.legend(*scatter.legend_elements(), loc="upper left")
    plt.title(f'[{method}] perplexity: {perplexity} iter: {itr}')
    plt.savefig(os.path.join(method, str(perplexity), f"{itr}.png"))


def plot_similarity(P, Q, perplexity, method):
    plt.clf()
    plt.title('[High Dimension] Similarity Distribution')
    plt.hist(x=P.flatten(), bins=100, log=True)
    plt.savefig(os.path.join(method, str(perplexity), "high.png"))

    plt.clf()
    plt.title('[Low Dimension] Similarity Distribution')
    plt.hist(x=Q.flatten(), bins=100, log=True)
    plt.savefig(os.path.join(method, str(perplexity), "low.png"))

if __name__ == '__main__':

    PERPLEXITY = 100
    METHOD = "T-SNE" # SSNE or T-SNE

    # load mnist dataset
    mninst_data = np.loadtxt('MNIST/mnist2500_X.txt')
    mnist_label = np.loadtxt('MNIST/mnist2500_labels.txt')

    # create a dir for this experiment
    os.makedirs(os.path.join(METHOD, str(PERPLEXITY)), exist_ok=True)

    # mninst_data shape = (2500, 784)
    # mninst_label = (784, )
    Y, P, Q = SNE(mninst_data, mnist_label, 2, 50, PERPLEXITY, METHOD)
    plot_similarity(P, Q, PERPLEXITY, METHOD)