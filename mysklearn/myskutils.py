import numpy as np
import csv
import random
import copy
from mysklearn import myutils as myutils
from mysklearn.myclassifiers import MyKNeighborsClassifier, MySimpleLinearRegressor, MyNaiveBayesClassifier
#from tabulate import tabulate

['Attribute', 'att1', 
    ['Value', 1, 
        ['Leaf', 8]
    ], 
    ['Value', 2, 
        ['Attribute', 'att2', 
            ['Value', 70.0, 
                ['Leaf', 6]], ['Value', 71.0, ['Leaf', 7]], ['Value', 72.0, ['Leaf', 5]], ['Value', 73.0, ['Leaf', 4]], ['Value', 74.0, ['Leaf', 6]], ['Value', 75.0, ['Leaf', 6]], ['Value', 76.0, ['Leaf', 7]], ['Value', 77.0, ['Leaf', 7]], ['Value', 78.0, ['Leaf', 7]], ['Value', 79.0, ['Leaf', 8]]]], ['Value', 3, ['Attribute', 'att2', ['Value', 70.0, ['Leaf', 5]], ['Value', 71.0, ['Leaf', 4]], ['Value', 72.0, ['Leaf', 5]], ['Value', 73.0, ['Leaf', 4]], ['Value', 74.0, ['Leaf', 4]], ['Value', 75.0, ['Leaf', 5]], ['Value', 76.0, ['Leaf', 6]], ['Value', 77.0, ['Leaf', 5]], ['Value', 78.0, ['Leaf', 5]], ['Value', 79.0, ['Leaf', 7]]]], ['Value', 4, ['Leaf', 4]], ['Value', 5, ['Attribute', 'att2', ['Value', 70.0, ['Leaf', 3]], ['Value', 71.0, ['Leaf', 1]], ['Value', 72.0, ['Leaf', 1]], ['Value', 73.0, ['Leaf', 1]], ['Value', 74.0, ['Leaf', 2]], ['Value', 75.0, ['Leaf', 3]], ['Value', 76.0, ['Leaf', 3]], ['Value', 77.0, ['Leaf', 3]], ['Value', 78.0, ['Leaf', 4]], ['Value', 79.0, ['Leaf', 4]]]]]
def try_lin_regress(LinRegress, table, y_train, X_train):
    """ trys linear regression with passed in values
    Args:
        LinRegress: (MyLinearRegressor): see myclassifiers
        table: (list of lists) table of items
        y_train: (list) y_train values
        X_train: (list of list) X_train values
    """
    print("===========================================")
    print("STEP 1: Linear Regression MPG Classifier")
    print("===========================================")

    test_instances = []
    rand_indexes = []

    #loops finds 5 random instances and adds X_train values
    for i in range(5):
        rand_index = random.randrange(0, len(X_train))
        rand_indexes.append(rand_index)
        test_instances.append(X_train[rand_index])
        X_train.remove(X_train[rand_index])
    
    #gets y predicted for text instances
    y_predicted = LinRegress.predict(test_instances)

    #loops though test instances and prints out y_predicted and y_actual
    for i in range(len(rand_indexes)):
        print("Instance: ", table[rand_indexes[i]])
        print("Class: ",y_predicted[i] , "Actual: " , y_train[rand_indexes[i]])

def try_KNN(Knn, table):
    """ trys Knn with passed in table
    Args:
        Knn: (MyKNeighborsClassifier): see myclassifiers
        table: (list of lists) table of items
    """
    print("===========================================")
    print("STEP 2: k=5 Nearest Neighbor MPG Classifier")
    print("===========================================")

    test_instances = []
    rand_indexes = []

    #loops finds 5 random instances and adds X_train values
    for i in range(5):
        rand_index = random.randrange(0, len(Knn.X_train))
        rand_indexes.append(rand_index)
        test_instances.append(Knn.X_train[rand_index])
        Knn.X_train.remove(Knn.X_train[rand_index])
    
    #gets y predicted for text instances
    y_predicted = Knn.predict(test_instances)

    #loops though test instances and prints out y_predicted and y_actual
    for i in range(len(rand_indexes)):
        print("Instance: ", table[rand_indexes[i]])
        print("Class: ",y_predicted[i] , "Actual: " , Knn.y_train[rand_indexes[i]])

def random_accuracy(X_train, X_test, y_train, y_test, lin_X_train, lin_X_test, lin_y_train, lin_y_test):
    """ Gets accuracies for Linear Regression and KNN classifiers for passed in values using train/test/split
    Args:
        X_train: (list of list) X_train for Knn classifier
        X_test: (list of list) X_tests for Knn classifier
        y_train: (list) y_train for Knn classifier
        y_test: (list) y_test to compare to for Linear Regression  classifier
        lin_X_train: (list of list) X_train for Linear Regression  classifier
        lin_X_test: (list of list) X_tests for Linear Regression  classifier
        lin_y_train: (list) y_train for Linear Regression classifier
        lin_y_test: (list) y_test to compare to for Linear Regression  classifier
    """

    #creates new linear regressor and KNN classifiers
    Knn = MyKNeighborsClassifier(n_neighbors=10)
    LinRegress = MySimpleLinearRegressor()

    #fits linear regressor and KNN classifier
    Knn.fit(X_train = X_train, y_train = y_train)
    LinRegress.fit(X_train = lin_X_train, y_train = lin_y_train)

    #gets predictions for Linear Regressor and Knn classifier
    y_predicted = Knn.predict(X_test)
    lin_y_predicted = LinRegress.predict(lin_X_test)
    
    #gets accuracys for Linear Regressor and Knn classifier
    k_acc = get_accuracy(y_predicted, y_test)
    lin_acc = get_accuracy(lin_y_predicted, lin_y_test)
    print("===========================================")
    print("STEP 3: Predictive Accuracy")
    print("===========================================")
    print("Random Subsample (k=10, 2:1 Train/Test)")
    print("Linear Regression: accuracy = ", lin_acc, " error rate = ", 1 - lin_acc)
    print("k Nearest Neighbors: accuracy = ", k_acc," error rate = ", 1 - k_acc)

def get_accuracy(y_predicted, y_test):
    """ gets accuracy for passed in y_predicted and y_actual
    Args:
        y_train: (list) y_train for Knn classifier or Linear Regression classifier
        y_test: (list) y_test to compare to for Knn classifier or Linear Regression  classifier

    returns: 
        acc: (float) accuracy of classifier 
    """
    acc_count = 0

    #loops though and checks for matches in paralel arrays
    for i in range(len(y_predicted)):
        if (y_predicted[i] == y_test[i]):
            acc_count+=1
    
    #calulates accuracy
    acc = acc_count/len(y_predicted)

    return acc

def perform_cross_validation(X_train2, X_train_folds, X_test_folds, y_train2, Classifier):
    """ performs cross validation on the passed in folds for Linear Regressor
    Args:
        X_train2: (list of list) initial X_train values
        X_train_folds (list of lists) folds of X_train indices
        y_train2: (list) initial y_train for Knn classifier
        X_test_folds (list of lists) folds of X_test indices
    
    returns: 
        y_predict: (list) y_predicted values
        y_test: (list) paralel list of y_actual values
    """
    X_train = []
    X_test = []
    y_predicted = []
    y_train = []
    y_test = []
    y_test = []
    y_predict = []
    curr_index = 0

    #loops though each X_test fold
    for i in range(len(X_test_folds)):
        X_test = []
        X_train = []
        y_train = []
        X_test = X_test_folds[i]

        #creates X test and y test from fold indices
        for j in range(len(X_test)):
            curr_index = X_test[j]
            X_test[j] = X_train2[curr_index]
            y_test.append(y_train2[curr_index])

        #creates X_train and inputs X_train items for each index
        X_train = X_train_folds[i]
        for j in range(len(X_train)):
            curr_index = X_train[j]
            X_train[j] = X_train2[curr_index]
            y_train.append(y_train2[curr_index])
        
        #tests Linear regression algorithm on each fold and appends values to y_predicted
        Classifier.fit(X_train = X_train, y_train = y_train)
        y_predicted.append(Classifier.predict(X_test))

    #converts y_predicted to 1d list
    for i in range(len(y_predicted)):
        for j in range(len(y_predicted[i])):
            y_predict.append(y_predicted[i][j])

    return y_predict, y_test

def perform_lin_cross_validation(X_train2, X_train_folds, X_test_folds, y_train2):
    """ performs cross validation on the passed in folds for KNN classifier
    Args:
        X_train2: (list of list) initial X_train values
        X_train_folds (list of lists) folds of X_train indices
        y_train2: (list) initial y_train for Knn classifier
        X_test_folds (list of lists) folds of X_test indices
    
    returns: 
        y_predict: (list) y_predicted values
        y_test: (list) paralel list of y_actual values
    """
    X_train = []
    X_test = []
    y_predicted = []
    y_train = []
    y_test = []
    y_test = []
    y_predict = []
    curr_index = 0

    #loops though each X_test fold
    for i in range(len(X_test_folds)):
        X_test = []
        X_train = []
        y_train = []
        X_test = X_test_folds[i]

        #creates X test and y test from fold indices
        for j in range(len(X_test)):
            curr_index = X_test[j]
            X_test[j] = X_train2[curr_index]
            y_test.append(y_train2[curr_index])

        #creates X_train and inputs X_train items for each index
        X_train = X_train_folds[i]
        for j in range(len(X_train)):
            curr_index = X_train[j]
            X_train[j] = X_train2[curr_index]
            y_train.append(y_train2[curr_index])
        
        #tests Linear regression algorithm on each fold and appends values to y_predicted
        lin = MySimpleLinearRegressor()
        lin.fit(X_train = X_train, y_train = y_train)
        y_predicted.append(lin.predict(X_test))

    #converts y_predicted to 1d list
    for i in range(len(y_predicted)):
        for j in range(len(y_predicted[i])):
            y_predict.append(y_predicted[i][j])

    return y_predict, y_test

def fold_accuracy(k_norm_y_predicted, k_norm_y_test, k_strat_y_predicted, k_strat_y_test, 
    lin_norm_y_predicted, lin_norm_y_test, lin_strat_y_predicted, lin_strat_y_test):
    """ gets accuracies for normal cross validation and stratified cross validation and prints out results
    Args:
        k_norm_y_predicted: (list) list of y predicted values for normal cross validation for KNN
        k_norm_y_test: list of y actual values for normal cross validation for KNN
        k_strat_y_predicted: list of y predicted values for stratified cross validation for KNN
        k_strat_y_test: list of y actual values for stratified cross validation for KNN
        lin_norm_y_predicted: (list) list of y predicted values for normal cross validation for KNN
        lin_norm_y_test: list of y actual values for normal cross validation for KNN
        lin_strat_y_predicted: list of y predicted values for stratified cross validation for KNN
        lin_strat_y_test: list of y actual values for stratified cross validation for KNN
    """

    k_norm_acc = get_accuracy(k_norm_y_predicted, k_norm_y_test)
    k_strat_acc = get_accuracy(k_strat_y_predicted, k_strat_y_test)

    lin_norm_acc = get_accuracy(lin_norm_y_predicted, lin_norm_y_test)
    lin_strat_acc = get_accuracy(lin_strat_y_predicted, lin_strat_y_test)

    print("===========================================")
    print("STEP 4: Predictive Accuracy")
    print("===========================================")
    print("10-Fold Cross Validation (k=10, 2:1 Train/Test)")
    print("Linear Regression: accuracy = " ,lin_norm_acc, " error rate = ", 1 - lin_norm_acc)
    print("k Nearest Neighbors: accuracy = ", k_norm_acc," error rate = ", 1 - k_norm_acc)
    print()
    print("10-Fold Stratified Cross Validation (k=10, 2:1 Train/Test)")
    print("Linear Regression: accuracy = " ,lin_strat_acc, " error rate = ", 1 - lin_strat_acc)
    print("k Nearest Neighbors: accuracy = ", k_strat_acc," error rate = ", 1 - k_strat_acc)

def print_matrix(matrix, labels,Class):
    """ prints confusion matrix
    Args:
        matrix: (list of lists): matrix of values
        labels: (list) x and y labels for matrix
        Class: (String) class label

    """
    count = 0
    header = []

    #setting up header by appending MPG, labels, Total, and Recognition
    header.append(Class)
    for item in labels:
        header.append(item)
    header.append("Total")
    header.append("Recognition")

    for i in range(len(matrix)):

        #gets counts for each actual label and appends total to matrix
        count = 0
        for j in range(len(matrix[i])):
            count += matrix[i][j]
        matrix[i].append(count)

        #checks for count of 0 to deny divison by 0
        if (count == 0):
            matrix[i].append(0)
        else:
            #appends accuracy for each label to matrix
            matrix[i].append((matrix[i][i] / count) * 100)

    #appends label to front of matrix
    for i in range(len(matrix)):
        matrix[i].insert(0,labels[i])
    
    print(tabulate(matrix, headers = header))

def try_Naive(Naive, table):
    """ trys Naive with passed in table
    Args:
        Naive: (MyKNeighborsClassifier): see myclassifiers
        table: (list of lists) table of items
    """
    print("===========================================")
    print("STEP 1: Naive Bayes MPG Classifier")
    print("===========================================")

    test_instances = []
    rand_indexes = []
    y_actual = []

    #loops finds 5 random instances and adds X_train values
    for i in range(5):
        rand_index = random.randrange(0, len(Naive.X_train))
        rand_indexes.append(rand_index)
        test_instances.append(Naive.X_train[rand_index])
        Naive.X_train.remove(Naive.X_train[rand_index])
    
    #gets y predicted for text instances
    y_predicted = Naive.predict(test_instances)
    myutils.convert_mpg_to_categorical(y_predicted)


    for i in range(len(rand_indexes)):
        y_actual.append(Naive.y_train[rand_indexes[i]])

    myutils.convert_mpg_to_categorical(y_actual)
    #loops though test instances and prints out y_predicted and y_actual
    for i in range(len(rand_indexes)):
        print("Instance: ", table[rand_indexes[i]])
        print("Class: ",y_predicted[i] , "Actual: " , y_actual[i])

def Naive_random_accuracy(X_train,X_test,y_train,y_test):
    """ Gets accuracies for Naive Bayes CLassifier for passed in values using train/test/split
    Args:
        X_train: (list of list) X_train for Naive Bayes classifier
        X_test: (list of list) X_tests for Naive Bayes classifier
        y_train: (list) y_train for Naive Bayes classifier
        y_test: (list) y_test to compare to for Naive Bayes classifier
      
    """

    #creates new linear regressor and KNN classifiers
    Naive = MyNaiveBayesClassifier()

    #fits linear regressor and KNN classifier
    Naive.fit(X_train = X_train, y_train = y_train)

    #gets predictions for Linear Regressor and Knn classifier
    y_predicted = Naive.predict(X_test)
    
    myutils.convert_mpg_to_categorical(y_predicted)
    myutils.convert_mpg_to_categorical(y_test)

    #gets accuracys for Linear Regressor and Knn classifier
    acc = get_accuracy(y_predicted, y_test)
    print("===========================================")
    print("STEP #2: Predictive Accuracy")
    print("===========================================")
    print("Random Subsample (k=10, 2:1 Train/Test)")
    print("Naive Bayes: accuracy = ", acc, " error rate = ", 1 - acc)

def perform_Naive_cross_validation(X_train2, X_train_folds, X_test_folds, y_train2):
    """ performs cross validation on the passed in folds for Naive Bayes
    Args:
        X_train2: (list of list) initial X_train values
        X_train_folds (list of lists) folds of X_train indices
        y_train2: (list) initial y_train for Knn classifier
        X_test_folds (list of lists) folds of X_test indices
    
    returns: 
        y_predict: (list) y_predicted values
        y_test: (list) paralel list of y_actual values
    """
    X_train = []
    X_test = []
    y_predicted = []
    y_train = []
    y_test = []
    y_test = []
    y_predict = []
    curr_index = 0

    #loops though each X_test fold
    for i in range(len(X_test_folds)):
        X_test = []
        X_train = []
        y_train = []
        X_test = X_test_folds[i]

        #creates X test and y test from fold indices
        for j in range(len(X_test)):
            curr_index = X_test[j]
            X_test[j] = X_train2[curr_index]
            y_test.append(y_train2[curr_index])

        #creates X_train and inputs X_train items for each index
        X_train = X_train_folds[i]
        for j in range(len(X_train)):
            curr_index = X_train[j]
            X_train[j] = X_train2[curr_index]
            y_train.append(y_train2[curr_index])
        
        #tests KNN algorithm on each fold and appends values to y_predicted
        Naive = MyNaiveBayesClassifier()
        Naive.fit(X_train = X_train, y_train = y_train)
        y_predicted.append(Naive.predict(X_test))

    #converts y_predicted to 1d list
    for i in range(len(y_predicted)):
        for j in range(len(y_predicted[i])):
            y_predict.append(y_predicted[i][j])

    return y_predict, y_test

def Naive_fold_accuracy(norm_y_predicted, norm_y_test, strat_y_predicted, strat_y_test):
    """ Gets accuracies for Naive Bayes Classifier
    Args:
        norm_y_predicted: (list) list of y predicted values for normal cross validation
        norm_y_test: list of y actual values for normal cross validation
        strat_y_predicted: list of y predicted values for stratified cross validation
        strat_y_test: list of y actual values for stratified cross validation
    """
    norm_acc = get_accuracy(norm_y_predicted, norm_y_test)
    strat_acc = get_accuracy(strat_y_predicted, strat_y_test)

    print("===========================================")
    print("STEP 3: Predictive Accuracy")
    print("===========================================")
    print("10-Fold Cross Validation (k=10, 2:1 Train/Test)")
    print("Naive Bayes: accuracy = " ,norm_acc, " error rate = ", 1 - norm_acc)
    print()
    print("10-Fold Stratified Cross Validation (k=10, 2:1 Train/Test)")
    print("Naive Bayes: accuracy = " ,strat_acc, " error rate = ", 1 - strat_acc)

def titanic_fold_accuracy(n_strat_y_predicted, n_strat_y_test, k_strat_y_predicted, k_strat_y_test, 
    z_strat_y_predicted, z_strat_y_test, r_strat_y_predicted, r_strat_y_test):
    """ gets accuracies for normal cross validation and stratified cross validation and prints out results
    Args:
        n_strat_y_predicted: (list) list of stratified y predicted values for Naive Bayes
        n_strat_y_test: list of y actual values for Stratified Naive Bayes
        k_strat_y_predicted: (list) list of stratified y predicted values for KNN
        k_strat_y_test: list of y actual values for Stratified KNN
        z_strat_y_predicted: (list) list of stratified y predicted values for ZeroR
        z_strat_y_test: list of y actual values for Stratified ZeroR
        r_strat_y_predicted: (list) list of stratified y predicted values for Random Classifer
        r_strat_y_test: list of y actual values for Stratified Random Classifier
    """
    n_strat_acc = get_accuracy(n_strat_y_predicted, n_strat_y_test)
    k_strat_acc = get_accuracy(k_strat_y_predicted, k_strat_y_test)
    z_strat_acc = get_accuracy(z_strat_y_predicted, z_strat_y_test)
    r_strat_acc = get_accuracy(r_strat_y_predicted, r_strat_y_test)


    print("===========================================")
    print("Naive Bayes and KNN Predictive Accuracy")
    print("===========================================")
    print("10-Fold Stratified Cross Validation (k=10, 2:1 Train/Test)")
    print("Naive Bayes: accuracy = " ,n_strat_acc, " error rate = ", 1 - n_strat_acc)
    print("KNN: accuracy = " ,k_strat_acc, " error rate = ", 1 - k_strat_acc)
    print("ZeroR Classifier: accuracy = " ,z_strat_acc, " error rate = ", 1 - z_strat_acc)
    print("Random Classifier: accuracy = " ,r_strat_acc, " error rate = ", 1 - r_strat_acc)


