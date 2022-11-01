from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from data_generator import polynomial_basis_linear_model

#%%

#%%

def create_design_matrix(data_x, num_basis):
    """
    data_x is a scalar, generate a design matrix based on 
    this scalar.

    design matrix = [x^0 x^1 x^2 ... x^(n-1)]
    """
    x = np.array(data_x)
    x = np.tile(x, (num_basis))
    power = np.arange(num_basis)
    design_matrix = np.power(x, power)
    return design_matrix.reshape((1, num_basis))

def linear_model_inference(xs, weight_mean, weight_variance=None, weight_variance_a=None, weight_variance_flag=None):
    xs = xs.reshape(-1, 1)
    xs = np.tile(xs, (weight_mean.shape[0]))
    power = np.arange(0, weight_mean.shape[0]).reshape(1, -1)
    power = np.tile(power, (xs.shape[0], 1))
    design_matrix = np.power(xs, power)
    ys = np.matmul(design_matrix, weight_mean)
    ys = np.squeeze(ys, axis=1)

    # add variance to ys
    if weight_variance is not None:
        ys_var = (1 / weight_variance_a) + np.matmul(np.matmul(design_matrix, np.linalg.inv(weight_variance)), design_matrix.T)
        ys_var = ys_var.diagonal()
        ys_var = ys_var * weight_variance_flag
        ys += ys_var

    return ys

def plot_result(xs, ys, color, axe, title, xlim, ylim):
    axe.plot(xs, ys, color=color)
    axe.set_title(title)

    if xlim != None:
        axe.set_xlim(xlim[0], xlim[1])

    if ylim != None:
        axe.set_ylim(ylim[0], ylim[1])


if __name__ == "__main__":
    
    # input for polynomial basis linear model (data generator)
    weight = []
    num_basis = int(input("num_basis: "))
    for i in range(num_basis):
        weight.append(float(input(f"weight {i}: ")))
    weight = np.array(weight).reshape((num_basis, 1))
    variance = float(input("variance: "))
    
    # input for bayesian linear regression
    a = 1 / variance
    b = float(input("b: "))

    # online learning for bayesian linear regression
    iter_idx = 0
    prior_cov, prior_mean = None, 0
    posterior_history = []
    data_xs, data_ys = [], []

    while True:
        iter_idx += 1

        # generate new sample from polynomial basis linear model
        data_x, data_y = polynomial_basis_linear_model(weight, variance)
        data_xs.append(data_x)
        data_ys.append(data_y)

        design_matrix = create_design_matrix(data_x, num_basis)

        # calculate covariance and mean of posterior
        
        # first iteration
        if iter_idx == 1:
            posterior_cov = a * np.matmul(design_matrix.T, design_matrix) + b * np.eye(num_basis)
            posterior_mean = a * np.matmul(np.linalg.inv(posterior_cov), design_matrix.T) * data_y
        
        # other iteration
        else:
            posterior_cov = a * np.matmul(design_matrix.T, design_matrix) + prior_cov
            tmp = (a * design_matrix.T * data_y) + np.matmul(prior_cov, prior_mean)
            posterior_mean = np.matmul(np.linalg.inv(posterior_cov), tmp)

        # store posterior information
        if iter_idx == 10 or iter_idx == 50:
            posterior_history.append([posterior_mean, posterior_cov])

        
        # marginalize mean and covariance for predictive distribution
        marginalize_mean = np.matmul(design_matrix, posterior_mean)[0][0]
        marginalize_cov = (1 / a) + np.matmul(np.matmul(design_matrix, np.linalg.inv(posterior_cov)), design_matrix.T)[0][0]

        # result of online learning in this iteration
        print(f"{'='*10} Iteration: {iter_idx} {'='*10}")
        print(f"Add data point ({data_x}, {data_y}):\n")
        print(f"posterior mean:")
        print(f"{posterior_mean}\n")
        print(f"posterior variance:")
        print(f"{np.linalg.inv(posterior_cov)}\n")
        print(f"predictive distribution ~ N({marginalize_mean}, {marginalize_cov})")

        # check if converge: posterior does not change a lot
        if np.linalg.norm(posterior_mean - prior_mean, ord=2) < 1e-4:
            posterior_history.append([posterior_mean, posterior_cov])
            break

        # update prior if not converge
        prior_mean = posterior_mean
        prior_cov = posterior_cov
        

    # visualization
    fig, axes = plt.subplots(nrows=2, ncols=2)

    data_x_min = min(data_xs) - 1
    data_x_max = max(data_xs) + 1
    xs = np.linspace(data_x_min, data_x_max, 5000)

    # groundtruth visualization ###############################################################################

    # plot mean
    ys = linear_model_inference(
        xs, 
        weight_mean=weight, 
    )
    plot_result(xs, ys, color="black", axe=axes[0][0], title="Ground Truth", xlim=None, ylim=(-20, 25))

    # plot one variance
    plot_result(xs, ys-variance, color="red", axe=axes[0][0], title="Ground Truth", xlim=None, ylim=(-20, 25))
    plot_result(xs, ys+variance, color="red", axe=axes[0][0], title="Ground Truth", xlim=None, ylim=(-20, 25))

    # final result visualization ###############################################################################

    # plot mean
    ys = linear_model_inference(
        xs, 
        weight_mean=posterior_history[2][0], 
    )
    plot_result(xs, ys, color="black", axe=axes[0][1], title="Predict Result", xlim=None, ylim=(-20, 25))

    # plot variance
    ys_pos_variance = linear_model_inference(
        xs, 
        weight_mean=posterior_history[2][0], 
        weight_variance=posterior_history[2][1], 
        weight_variance_a=a,
        weight_variance_flag=1
    )
    ys_neg_variance = linear_model_inference(
        xs, 
        weight_mean=posterior_history[2][0], 
        weight_variance=posterior_history[2][1], 
        weight_variance_a=a,
        weight_variance_flag=-1
    )
    plot_result(xs, ys_pos_variance, color="red", axe=axes[0][1], title="Predict Result", xlim=None, ylim=(-20, 25))
    plot_result(xs, ys_neg_variance, color="red", axe=axes[0][1], title="Predict Result", xlim=None, ylim=(-20, 25))
    axes[0][1].scatter(data_xs, data_ys, color="blue", s=0.5)

    # 10 data points result visualization ###############################################################################
    
    # plot mean
    ys = linear_model_inference(
        xs, 
        weight_mean=posterior_history[0][0], 
    )
    plot_result(xs, ys, color="black", axe=axes[1][0], title="After 10 Incomes", xlim=None, ylim=(-20, 25))
    
    # plot variance
    ys_pos_variance = linear_model_inference(
        xs,
        weight_mean=posterior_history[0][0], 
        weight_variance=posterior_history[0][1], 
        weight_variance_a=a,
        weight_variance_flag=1
    )
    ys_neg_variance = linear_model_inference(
        xs, 
        weight_mean=posterior_history[0][0], 
        weight_variance=posterior_history[0][1], 
        weight_variance_a=a,
        weight_variance_flag=-1
    )
    plot_result(xs, ys_pos_variance, color="red", axe=axes[1][0], title="After 10 Incomes", xlim=None, ylim=(-20, 25))
    plot_result(xs, ys_neg_variance, color="red", axe=axes[1][0], title="After 10 Incomes", xlim=None, ylim=(-20, 25))
    axes[1][0].scatter(data_xs[:10], data_ys[:10], color="blue", s=0.5)

    # 50 data points result visualization ###############################################################################
    
    # plot mean
    ys = linear_model_inference(
        xs,
        weight_mean=posterior_history[1][0], 
    )
    plot_result(xs, ys, color="black", axe=axes[1][1], title="After 50 Incomes", xlim=None, ylim=(-20, 25))
    
    # plot variance
    ys_pos_variance = linear_model_inference(
        xs,
        weight_mean=posterior_history[1][0], 
        weight_variance=posterior_history[1][1], 
        weight_variance_a=a,
        weight_variance_flag=1
    )
    ys_neg_variance = linear_model_inference(
        xs, 
        weight_mean=posterior_history[1][0], 
        weight_variance=posterior_history[1][1], 
        weight_variance_a=a,
        weight_variance_flag=-1
    )
    plot_result(xs, ys_pos_variance, color="red", axe=axes[1][1], title="After 50 Incomes", xlim=None, ylim=(-20, 25))
    plot_result(xs, ys_neg_variance, color="red", axe=axes[1][1], title="After 50 Incomes", xlim=None, ylim=(-20, 25))
    axes[1][1].scatter(data_xs[:50], data_ys[:50], color="blue", s=0.5)

    plt.show()