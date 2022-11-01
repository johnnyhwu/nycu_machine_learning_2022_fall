import numpy as np

def univariate_gaussian(mean, variance):
    points = np.random.uniform(0.0, 1.0, 12)
    total_sum = np.sum(points)
    standard_normal = total_sum - 6
    return standard_normal * np.sqrt(variance) + mean

def polynomial_basis_linear_model(weight, variance):
    x = np.random.uniform(-1.0, 1.0, 1)
    uniform_x = np.repeat(a=x, repeats=weight.shape[0], axis=0)
    exponential = np.arange(0, weight.shape[0])
    basis = np.power(uniform_x, exponential).reshape(1, -1)
    y = np.matmul(basis, weight) + univariate_gaussian(0, variance)
    return x[0], y[0][0]


if __name__ == "__main__":
    is_gaussian = True

    if is_gaussian:
        mean = float(input("mean: "))
        variance = float(input("variance: "))
        value = univariate_gaussian(mean, variance)
        print(value)
    else:
        weight = []
        num_basis = int(input("num_basis: "))
        for i in range(num_basis):
            weight.append(float(input(f"weight {i}: ")))
        weight = np.array(weight).reshape((num_basis, 1))
        variance = float(input("variance: "))
        value_x, value_y = polynomial_basis_linear_model(weight, variance)
        print(f"({value_x}, {value_y})")