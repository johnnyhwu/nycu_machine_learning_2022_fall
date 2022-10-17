import numpy as np

def load_dataset(mode: str) -> tuple:
    if mode == "training":
        img_path = "dataset/train-images-idx3-ubyte"
        label_path = "dataset/train-labels-idx1-ubyte"
    else:
        img_path = "dataset/t10k-images-idx3-ubyte"
        label_path = "dataset/t10k-labels-idx1-ubyte"
    
    # read image meta data and image pixel value
    with open(img_path, "rb") as f:
        _, num_img, num_row, num_col = np.fromfile(file=f, dtype=">i4", count=4)
        all_imgs = np.fromfile(file=f, dtype=">B", count=-1)
    
    # read image label
    with open(label_path, "rb") as f:
        _ = np.fromfile(file=f, dtype=">i4", count=2)
        all_labels = np.fromfile(file=f, dtype=">B", count=-1)
    
    # reshape image to Nx28x28
    all_imgs = np.reshape(all_imgs, (num_img, num_row, num_col))

    return (all_imgs, all_labels)


if __name__ == "__main__":
    (all_imgs, all_labels) = load_dataset(mode="training")
    print(all_imgs.shape)
    print(all_labels.shape)