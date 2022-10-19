from collections import Counter
from math import factorial

def load_data(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def binomial_likelihood(m, N, theta):
    # combination term
    comb = factorial(N) / (factorial(m) * factorial(N-m))

    # probability of success in m times
    prob_success = theta ** m

    # probability of failure in N-m times
    prob_failure = (1-theta) ** (N-m)

    return comb * prob_success * prob_failure

if __name__ == "__main__":
    # a series of outcomes
    outcomes = load_data("testfile.txt")

    # initial a = alpha-1 and b = beta-1 paramters of beta distribution
    # alpha and beta are paramters of beta distribution
    # we can simply think "a" as number of success and "b" as number of failure
    alpha_1 = int(input("a (alpha-1 = number of success): "))
    beta_1 = int(input("b (beta-1 = number of failure): "))
    print()

    # initial prior
    prior = (alpha_1, beta_1)

    # iterate all outcomes
    for idx, outcome in enumerate(outcomes):
        outcome = Counter(outcome)
        num_success = outcome['1']
        num_failure = outcome['0']
        
        # calculate likelihood based on current outcome
        likelihood = binomial_likelihood(
            m=num_success,
            N=num_success+num_failure,
            theta=num_success/(num_success+num_failure)
        )

        # calculate posterior based on likelihood and prior
        # represent posterior in bea distribution
        posterior = (num_success+prior[0], num_failure+prior[1])

        print(f"case {idx+1}: {outcome}")
        print(f"Likelihood: {likelihood}")
        print(f"Prior: {prior}")
        print(f"Posterior: {posterior}")
        print()

        # online learning
        prior = posterior