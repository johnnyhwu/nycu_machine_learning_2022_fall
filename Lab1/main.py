import solver
from matplotlib import pyplot as plt

def user_input():
    num_basis = int(input('Number of basis: '))
    reg_term = int(input('Lambda (used in LSE): '))
    return num_basis, reg_term

def read_dataset():
    with open('testfile.txt', 'r') as f:
        texts = f.readlines()
    xs = [float(text.strip().split(',')[0]) for text in texts]
    ys = [float(text.strip().split(',')[1]) for text in texts]
    return xs, ys

def print_matrix(name, matrix):
    print(f"{name}: ")
    for row in matrix:
        print(row)
    print()

def print_fitting_line(mat):
    result = ""
    for idx in range(0, len(mat)):
        if idx > 0:
            result += " + "
        result += f"{mat[idx][0]}x^{idx}"
    print(f"Fitting Line: {result} = y")




if __name__ == "__main__":
    xs, ys = read_dataset()
    num_basis, reg_term = user_input()
    print('-'*10)
    
    # test ssample:
    # xs = [1.0, 2.0, 3.0]
    # ys = [-1.0, 5.0, 2.0]
    # num_basis = 3
    # reg_term = 0

    plt.subplot(2, 1, 1)
    
    print("Least Square Error: ")
    mat1, (error, ys_pred) = solver.least_square_error(xs, ys, num_basis, reg_term)
    print_fitting_line(mat1)
    print(f"Total Error: {error}")

    plt.subplot(2, 1, 1)
    plt.plot(xs, ys_pred, c='blue', label='Least Square Error')
    plt.scatter(xs, ys, c='red')
    plt.title('Least Square Error')
    plt.ylabel('y Axis')
    plt.legend()

    print()

    print("Newton Method: ")
    mat2, (error, ys_pred) = solver.newton_method(xs, ys, num_basis)
    print_fitting_line(mat2)
    print(f"Total Error: {error}")

    plt.subplot(2, 1, 2)
    plt.plot(xs, ys_pred, c='blue', label='Newton Method')
    plt.scatter(xs, ys, c='red')
    plt.xlabel('x Axis')
    plt.ylabel('y Axis')
    plt.legend()

    plt.show()
    