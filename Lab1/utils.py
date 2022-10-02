def gauss_jordan(mat):
    rank = len(mat)

    identity_mat = []
    for i in range(rank):
        row = [0]*i + [1]*1 + [0]*(rank-(i+1))
        identity_mat.append(row)

    # below diagonal
    for c in range(rank-1):
        for r in range(c+1, rank):
            assert mat[c][c] != 0, f"Diagonal value is zero at ({c}, {c})"
            diff = -(mat[r][c] / mat[c][c])
            for j in range(rank):
                mat[r][j] += diff*mat[c][j]
                identity_mat[r][j] += diff*identity_mat[c][j]
    
    # diagonal
    for r in range(rank):
        diag = mat[r][r]
        for j in range(rank):
            mat[r][j] /= diag
            identity_mat[r][j] /= diag
    
    # above diagonal
    for c in range(rank):
        for r in range(0, c):
            diff = -(mat[r][c] / mat[c][c])
            for j in range(rank):
                mat[r][j] += diff*mat[c][j]
                identity_mat[r][j] += diff*identity_mat[c][j]
    
    return identity_mat


def build_mat_A(xs, num_basis):
    return [[x**i for i in range(num_basis)] for x in xs]

def build_mat_b(ys):
    return [[val] for val in ys]

def build_mat_identity(rank):
    res = []
    for i in range(rank):
        res.append([0.0]*i + [1.0]*1 + [0.0]*(rank-(i+1)))
    return res

def transpose_mat(mat):
    res = []
    for c in range(len(mat[0])):
        row = []
        for r in range(len(mat)):
            row.append(mat[r][c])
        res.append(row)
    return res

def multiply_mat_mat(mat1, mat2):
    mat1_rows, mat1_cols, mat2_cols = len(mat1), len(mat1[0]), len(mat2[0])
    res = []
    for r in range(mat1_rows):
        new_row = []
        for c in range(mat2_cols):
            sum = 0
            for j in range(mat1_cols):
                sum += mat1[r][j] * mat2[j][c]
            new_row.append(sum)
        res.append(new_row)
    return res

def multiply_scalar_mat(scalar, mat):
    res = []
    for r in range(len(mat)):
        row = []
        for c in range(len(mat[0])):
            row.append(scalar * mat[r][c])
        res.append(row)
    return res

def add_mat_mat(mat1, mat2):
    res = []
    for r in range(len(mat1)):
        row = []
        for c in range(len(mat1[0])):
            row.append(mat1[r][c] + mat2[r][c])
        res.append(row)
    return res

def sub_mat_mat(mat1, mat2):
    res = []
    for r in range(len(mat1)):
        row = []
        for c in range(len(mat1[0])):
            row.append(mat1[r][c] - mat2[r][c])
        res.append(row)
    return res


if __name__ == "__main__":
    # mat = [
    #     [1.0, 1.0],
    #     [2.0, -1.0]
    # ]

    mat1 = [
        [2, 1, -1],
        [1, -3, 1],
        [1, 3, -3]
    ]

    mat2 = [
        [2, 3],
        [-1, 5],
        [4, 0]
    ]

    mat3 = [
        [0, -1, 5],
        [1, 1, 1],
        [-3, 2, 4]
    ]

    ys = [1, 2, 3]

    # print(gauss_jordan(mat))
    # print(transpose_mat(mat))
    # print(multiply_mat(mat1, mat2))
    # print(build_mat_identity(5))
    # print(multiply_scalar_mat(5, mat1))
    # print(add_mat_mat(mat1, mat3))
    # print(build_mat_b(ys))
    print(sub_mat_mat(mat1, mat3))