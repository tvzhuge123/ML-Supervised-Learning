from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def decisionTree(data):
    # find the best alpha for pruning
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.5, random_state=0)
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-2]

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Alpha")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    # best alpha
    ccp_alpha = ccp_alphas[test_scores.index(max(test_scores))]
    print('Alpha: %2f' % ccp_alpha)

    # train with various training size
    train_size = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0,1,2,3,4,5,6,7,8,9, 10,11, 12, 13, 14, 100, 1000]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1-ts, random_state=state)

            # train and evaluate model
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1)/len(tmp1))
        test_scores.append(sum(tmp2)/len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(print('Overall accuracy: %2f' % max(test_scores)))


def neuralNetwork(data):
    # train with various training size
    train_size = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0,1,2,3]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1-ts, random_state=state)

            # train and evaluate model
            clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                                hidden_layer_sizes=(100, 5), random_state=1, max_iter=10000)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1)/len(tmp1))
        test_scores.append(sum(tmp2)/len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(max(test_scores))


def boosting(data):
    # find the best n iterations
    n_list = [1, 2, 5, 10, 20, 50, 100]
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.5, random_state=0)
    clfs = []
    for n in n_list:
        clf = GradientBoostingClassifier(n_estimators=n, learning_rate=1.0,  max_depth = 1, random_state = 0)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("# of iteratoins")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs # of iteratoins")
    ax.plot(n_list, train_scores, marker='o', label="train")
    ax.plot(n_list, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    # best alpha
    n = n_list[test_scores.index(max(test_scores))]
    print('# of iteratoins: %s' % n)

    # train with various training size
    train_size = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0, 1, 2]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1 - ts, random_state=state)

            # train and evaluate model
            clf = GradientBoostingClassifier(n_estimators=n, learning_rate=1.0,  max_depth = 1, random_state = 0)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1) / len(tmp1))
        test_scores.append(sum(tmp2) / len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(print('Overall accuracy: %2f' % max(test_scores)))


def supportVectorMachine(data, kernel):
    # find the best C
    c_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 10000, 50000]
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.5, random_state=0)
    clfs = []
    for c in c_list:
        clf = svm.SVC(kernel=kernel, C=c)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs C")
    ax.plot(c_list, train_scores, marker='o', label="train")
    ax.plot(c_list, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    # best alpha
    c = c_list[test_scores.index(max(test_scores))]
    print('C: %s' % c)

    # train with various training size
    train_size = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0,1,2,3]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1 - ts, random_state=state)

            # train and evaluate model
            clf = svm.SVC(kernel=kernel, C=c)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1) / len(tmp1))
        test_scores.append(sum(tmp2) / len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(print('Overall accuracy: %2f' % max(test_scores)))


def kNearestNeighbors(data):
    # find the best n
    n_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.5, random_state=0)
    clfs = []
    for n in n_list:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("# of neighbors")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs # of neighbors")
    ax.plot(n_list, train_scores, marker='o', label="train")
    ax.plot(n_list, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    # best alpha
    n = n_list[test_scores.index(max(test_scores))]
    print('# of neighbors: %s' % n)

    # train with various training size
    train_size = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0,1,2,3,4,5,6,7,8,9, 10,11, 12, 13, 14, 100, 1000]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1 - ts, random_state=state)

            # train and evaluate model
            clf = KNeighborsClassifier(n_neighbors=n)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1) / len(tmp1))
        test_scores.append(sum(tmp2) / len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(print('Overall accuracy: %2f' % max(test_scores)))


def evaluation(clf, X_train, X_test, y_train, y_test):
    # evaluate model by cross validation
    res_train = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    res_test = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')

    return sum(res_train)/len(res_train), sum(res_test)/len(res_test)



def main():
    data_wine = load_wine(return_X_y=True)
    data_bc = load_breast_cancer(return_X_y=True)

    decisionTree(data_wine)
    decisionTree(data_bc)

    neuralNetwork(data_wine)
    neuralNetwork(data_bc)

    boosting(data_wine)
    boosting(data_bc)

    supportVectorMachine(data_wine, 'linear')
    supportVectorMachine(data_wine, 'rbf')
    supportVectorMachine(data_bc, 'poly')
    supportVectorMachine(data_bc, 'rbf')

    kNearestNeighbors(data_wine)
    kNearestNeighbors(data_bc)


if __name__ == '__main__':
    main()