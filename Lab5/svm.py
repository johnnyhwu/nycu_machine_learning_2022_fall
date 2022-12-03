import numpy as np
import libsvm.svmutil as svm
from multiprocessing import Process, Value, Array


def read_dataset(path="data"):

    # X_train
    with open(f"{path}/X_train.csv", "r") as f:
        contents = f.readlines()
    X_train = [list(map(float, text.strip().split(","))) for text in contents]
    X_train = np.array(X_train, dtype=np.float32)

    # y_train
    with open(f"{path}/Y_train.csv", "r") as f:
        contents = f.readlines()
    y_train = [[int(text.strip())] for text in contents]
    y_train = np.array(y_train, dtype=np.int8).flatten()

    # X_test
    with open(f"{path}/X_test.csv", "r") as f:
        contents = f.readlines()
    X_test = [list(map(float, text.strip().split(","))) for text in contents]
    X_test = np.array(X_test, dtype=np.float32)

    # y_test
    with open(f"{path}/Y_test.csv", "r") as f:
        contents = f.readlines()
    y_test = [[int(text.strip())] for text in contents]
    y_test = np.array(y_test, dtype=np.int8).flatten()

    return X_train, y_train, X_test, y_test


def grid_search(X_train, y_train, all_combinations, best_hyperparams, best_acc):

    for idx, params in enumerate(all_combinations):
        print(f"#{idx}: ", end="")
        train_acc = svm.svm_train(y_train, X_train, params)
        if train_acc > best_acc.value:
            with best_acc.get_lock() and best_hyperparams.get_lock():
                best_acc.value = train_acc
                for i in range(100):
                    best_hyperparams[i] = 32
                for i in range(len(params)):
                    best_hyperparams[i] = ord(params[i])


if __name__ == "__main__":

    EXECUTION_PART = 2

    # read dataset
    X_train, y_train, X_test, y_test = read_dataset()

    if EXECUTION_PART == 1:

        # part 1: linear kernel, polynomial kernel and RBF kernel in SVM
        # linear kernel
        print("Linear Kernel: ", end="")
        model = svm.svm_train(y_train, X_train, f"-t 0 -q")
        pred_label, pred_acc, pred_val = svm.svm_predict(y_test, X_test, model)

        # polynomial
        print("Polynomial Kernel: ", end="")
        model = svm.svm_train(y_train, X_train, f"-t 1 -q")
        pred_label, pred_acc, pred_val = svm.svm_predict(y_test, X_test, model)

        # polynomial
        print("RBF Kernel: ", end="")
        model = svm.svm_train(y_train, X_train, f"-t 2 -q")
        pred_label, pred_acc, pred_val = svm.svm_predict(y_test, X_test, model)
    
    elif EXECUTION_PART == 2:

        # part 2: find best hyperparameters for C-SVC with linear, polynomial and RBF kernel
        hyperparams = {
            'degree': [1, 2, 3, 4, 5], # polynomail kernel (default = 3)
            'gamma': [
                0.00127551*(1/4),
                0.00127551*(1/2),
                0.00127551*1,
                0.00127551*2,
                0.00127551*4,
                0.00127551*8,
                0.00127551*16,
                0.00127551*32, 
                0.00127551*64,
                0.00127551*128,
            ],  # polynomail and RBF kernel (default = 1/784 = 0.00127551)
            'coef0': [0, 1, 2, 3, 4], # polynomail kernel (default = 0)
            'cost': [0.01, 0.1, 1, 10, 25, 50 ,75, 100, 200], # C-SVC (default = 1)
        }

        # 2349 total

        all_combinations = []

        for kernel_idx, _ in enumerate(["linear", "polynomial", "rbf"]):

            # linear kernel
            if kernel_idx == 0:
                for c in hyperparams["cost"]:
                    params = f"-t {kernel_idx} -c {c} -v 3 -q"
                    all_combinations.append(params)
            
            # polynomail kernel
            elif kernel_idx == 1:
                for c in hyperparams["cost"]:
                    for g in hyperparams["gamma"]:
                        for d in hyperparams["degree"]:
                            for co in hyperparams["coef0"]:
                                params = f"-t {kernel_idx} -c {c} -g {g} -d {d} -r {co} -v 3 -q"
                                all_combinations.append(params)
            
            # RBF kernel
            else:
                for c in hyperparams["cost"]:
                    for g in hyperparams["gamma"]:
                        params = f"-t {kernel_idx} -c {c} -g {g} -v 3 -q"   
                        all_combinations.append(params)

        print(f"Total number of combinations: {len(all_combinations)}")

        process_tasks = int(len(all_combinations) // 8)
        process_pool = []
        best_hyperparams = Array('i', 100)
        best_acc = Value('d', 0)

        for i in range(8):
            
            start_idx = i * 8
            end_idx = (i+1) * 8
            if end_idx == 7:
                end_idx = len(all_combinations)
            
            p = Process(
                    target=grid_search, 
                    args=(
                        X_train, 
                        y_train, 
                        all_combinations[start_idx:end_idx],
                        best_hyperparams,
                        best_acc
                    )
                )
            
            process_pool.append(p)
        
        for p in process_pool:
            p.start()
        
        for p in process_pool:
            p.join()

        print("best_hyperparams: ")
        for i in range(len(best_hyperparams)):
            print(chr(best_hyperparams[i]), end="")
        print()
        print(f"best_acc: {best_acc.value}")

