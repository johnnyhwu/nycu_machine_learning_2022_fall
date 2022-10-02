import utils

def infer_model(mat, x):
    sum = 0
    for idx in range(0, len(mat)):
        sum += mat[idx][0] * x**idx
    return sum


def total_error(mat, xs, ys):
    sum = 0
    ys_pred = []
    for (x, y) in zip(xs, ys):
        y_pred = infer_model(mat, x)
        ys_pred.append(y_pred)

        sum += (y_pred - y)**2
    return (sum, ys_pred)


def least_square_error(xs, ys, num_basis, reg_term):
    mat_b = utils.build_mat_b(ys)
    mat_A = utils.build_mat_A(xs, num_basis)

    mat_AT = utils.transpose_mat(mat_A)
    mat_ATA = utils.multiply_mat_mat(mat_AT, mat_A)
    mat_reg = utils.multiply_scalar_mat(reg_term, utils.build_mat_identity(len(mat_ATA)))
    mat = utils.add_mat_mat(mat_ATA, mat_reg)
    mat_inv = utils.gauss_jordan(mat)

    mat = utils.multiply_mat_mat(mat_inv, mat_AT)
    mat = utils.multiply_mat_mat(mat, mat_b)
    return mat, total_error(mat, xs, ys), 

def newton_method(xs, ys, num_basis):
    mat_b = utils.build_mat_b(ys)
    mat_A = utils.build_mat_A(xs, num_basis)

    # initial x=0
    mat_x = [[0] for _ in range(len(mat_A[0]))]

    while True:
    
        # gradient
        mat_AT = utils.transpose_mat(mat_A)
        mat_ATA = utils.multiply_mat_mat(mat_AT, mat_A)
        mat_ATAx = utils.multiply_mat_mat(mat_ATA, mat_x)
        mat_ATb = utils.multiply_mat_mat(mat_AT, mat_b)
        gradient = utils.multiply_scalar_mat(2, utils.sub_mat_mat(mat_ATAx, mat_ATb))

        # check gradient
        gradient_sum = 0
        for val in gradient:
            gradient_sum += abs(val[0])
        if gradient_sum < 1e-5:
            break

        # hessian
        hessian = utils.multiply_scalar_mat(2, mat_ATA)
        hessian_inv = utils.gauss_jordan(hessian)

        # update x
        mat_x = utils.sub_mat_mat(mat_x, utils.multiply_mat_mat(hessian_inv, gradient))
    
    return mat_x, total_error(mat_x, xs, ys)
