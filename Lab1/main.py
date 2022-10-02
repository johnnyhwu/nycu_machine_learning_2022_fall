import solver

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


if __name__ == "__main__":
    xs, ys = read_dataset()
    num_basis, reg_term = user_input()

    # test ssample:
    # xs = [1.0, 2.0, 3.0]
    # ys = [-1.0, 5.0, 2.0]
    # num_basis = 3
    # reg_term = 0

    mat1 = solver.least_square_error(xs, ys, num_basis, reg_term)
    print(mat1)

    mat2 = solver.newton_method(xs, ys, num_basis)
    print(mat2)
    