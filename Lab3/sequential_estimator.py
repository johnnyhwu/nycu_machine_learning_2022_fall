import data_generator as dg

if __name__ == "__main__":
    threshold = 0.01
    target_mean = float(input("mean: "))
    target_variance = float(input("variance: "))

    # initialize
    sample_mean, sample_var, m2, count = 0, 0, 0, 0

    # Welford's online algorithm
    while True:
        data = dg.univariate_gaussian(target_mean, target_variance)
        count += 1
        
        delta_1 = data - sample_mean
        sample_mean += delta_1 / count
        delta_2 = data - sample_mean
        m2 += delta_1 * delta_2
        sample_var = m2 / count

        print(f"Iteration: {count}")
        print(f"Add data point: {data}")
        print(f"Mean = {sample_mean} Variance = {sample_var}")
        print()

        # converge
        if abs(sample_mean - target_mean) <= threshold and abs(sample_var - target_variance) <= threshold:
            break