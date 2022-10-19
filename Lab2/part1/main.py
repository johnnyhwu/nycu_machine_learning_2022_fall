import utils
import solver

if __name__ == "__main__":
    
    # load training and testing dataset
    (train_imgs, train_labels) = utils.load_dataset(mode="training")
    (test_imgs, test_labels) = utils.load_dataset(mode="testing")

    # naive bayes classifier
    classfier = solver.NaiveBayesClassifier(
        num_class=10,
        num_feature=28*28,
        dist_type="continuous" # continuous or discrete
    )
    classfier.fit(
        X=train_imgs,
        y=train_labels
    )
    classfier.distill()
    classfier.evaluate(
        X=test_imgs,
        y=test_labels,
        show_info=False
    )
    