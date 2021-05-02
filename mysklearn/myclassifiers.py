import mysklearn.myutils as myutils
import operator
import copy
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        
            """
        #getting mean for X_train
        x_sum = 0
        for i in range(len(X_train)):
            x_sum += (X_train[i][0])
        mean_x = x_sum / len(X_train)
            
        #gettting mean for y_train
        y_sum = 0
        for i in range(len(y_train)):
            y_sum += (y_train[i])
        mean_y = y_sum / len(y_train)

        #computing slope and y intercept
        self.slope = sum([(X_train[i][0] - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train))]) \
            / sum([(X_train[i][0] - mean_x) ** 2 for i in range(len(X_train))])
        # y = mx + b => y - mx
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for i in range(len(X_test)):
            y_predicted.append(int(round((X_test[i][0] * self.slope) + self.intercept)))

        #for use of mpg regressor, remove for other functions
        for i in range(len(y_predicted)):
            if (y_predicted[i] == 0):
                y_predicted[i] = 1
            
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        indices = []
        tot_indices = []
        tot_distances = []
        Xlen = len(X_test[0])
     
        for j in range(len(X_test)):
            for i, instance in enumerate(self.X_train):
                # append the label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                dist = myutils.compute_euclidean_distance(instance[:len(instance)-2], X_test[j][0:Xlen])
                instance.append(dist)
            
            #sorting training set and getting k nearest neighbors
            train_sorted = sorted(self.X_train, key=operator.itemgetter(-1))
            top_k = train_sorted[:self.n_neighbors]
            
            distances = []
            indices = []

            #appending distances and indeces to array
            for i in range(len(top_k)):
                distances.append(top_k[i][-1])
                indices.append(top_k[i][-2])
            tot_distances.append(distances)
            tot_indices.append(indices)

            # removing distance, index and y_train from X_train
            for instance in self.X_train:
                del instance[-3:]

        return tot_distances, tot_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, indices = self.kneighbors(X_test)
        neighbors = []
        y_predicted = []

        #loops through each X_test
        for i in range(len(X_test)):
            neighbors = []

            #loops through indices and adds y_train values to neighbors
            for j in range(len(indices[i])):
                neighbors.append(self.y_train[indices[i][j]])
        
            #gets frequencies of neighbors and y value with highest count to set as y predicted
            values, counts = myutils.get_frequencies(neighbors)
            max_val = max(counts)
            max_index = counts.index(max_val)
            prediction = values[max_index]  
            y_predicted.append(prediction)  

        #returns array of y-predicted values
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = [] 
        self.posteriors = []

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        row = []
        matrix = []
        classes = []
        posteriors = []
        labels = []
        count = 0
        self.X_train = X_train
        self.y_train = y_train

        labels.append("Attributes")
        for i in range(len(y_train)):
            if y_train[i] not in labels:
                labels.append(y_train[i])
                classes.append(y_train[i])
        
        #calculating priors
        for i in range(len(classes)):
            count = 0
            for j in range(len(y_train)):
                if (classes[i] == y_train[j]):
                    count+=1
            self.priors.append(count/len(y_train))

        for i in range(len(X_train[0])):
            curr_col = myutils.get_column(X_train,i)
            matrix = []
            matrix.append(labels)
            items = []
            for j in range(len(curr_col)):
                if (curr_col[j] not in items):
                    curr_row = []
                    currAtt = curr_col[j]
                    items.append(currAtt)
                    curr_row.append(currAtt)
                    for k in range(len(classes)):
                        count = 0
                        for l in range(len(curr_col)):
                            if curr_col[l] == currAtt and y_train[l] == classes[k]:
                                count+=1
                        curr_row.append( round((count/len(curr_col)) / self.priors[k],3))
                    matrix.append(curr_row)
            self.posteriors.append(matrix)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        probabilities = []
        y_predicted = []
        for i in range(len(X_test)):
            probabilities = []
            for m in range(len(self.priors)):
                prob = 1
                for k in range(len(self.posteriors)):
                    currMatrix = self.posteriors[k]
                    currCol = myutils.get_column(currMatrix,m+1)
                    for j in range(len(currMatrix)):
                        if (currMatrix[j][0] == X_test[i][k]):
                            prob = prob * currCol[j]
                probabilities.append(prob * self.priors[m])            
            maxProb = probabilities.index(max(probabilities))
            y_predicted.append(self.posteriors[0][0][maxProb +1])

        return y_predicted 

class MyZeroRClassifier:
    """Represents a ZeroR classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """

        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        mode = max(set(self.y_train), key=self.y_train.count)
        for i in range(len(X_test)):
            y_predicted.append(mode)
        return y_predicted

class MyRandomClassifier:
    """Represents a Random classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """

        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        weights = []
        values, counts = myutils.get_frequencies(self.y_train)
        for i in range(len(self.y_train)):
            for j in range(len(values)):
                if (self.y_train[i] == values[j]):
                    weights.append(counts[j])
        y_predicted = random.choices(self.y_train, weights=weights, k=len(X_test))
    
        return y_predicted # TODO: fix this

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
         # fit() accepts X_train and y_train
        # TODO: calculate the attribute domains dictionary

        header = []
        attribute_domains = {}
        
        #loops through X_train and creates header
        for i in range(len(X_train[0])) :
            header.append("att" + str(i))

        #loops though header to form attribute domains dictionairy
        count = 0
        for item in header:
            curr_col = myutils.get_column(X_train, count)
            values, counts = myutils.get_frequencies(curr_col)
            attribute_domains[item] = values
            count+=1

        #stitching together X_train and y_train and getting available attributes
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy()
    
        #forming tree
        self.tree = myutils.tdidt(train, available_attributes, attribute_domains, header)
        self.print_decision_rules()

        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for item in X_test:
            y_predict = myutils.predict_helper(item,self.tree)
            y_predicted.append(y_predict[1])

        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """

        rule = []
        myutils.print_tree_helper(self.tree, rule, "", attribute_names, class_name)
        

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    """Represents a decision tree classifier.

    Attributes:
        N(int): Number of Classifiers to develop
        M(int): Number of better Classifiiers to use
        F(int): Size of random attribute subset
        X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        trees(list of nested lists): list of trees generated

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, n = 20, m = 7, f = 2, seed = None):
        """Initializer for MyDecisionTreeClassifier.

        """

        self.N = n
        self.M = m
        self.F = f
        self.X_train = None 
        self.y_train = None
        self.trees = []
        self.seed = seed

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
         # fit() accepts X_train and y_train
        # TODO: calculate the attribute domains dictionary

        if (self.seed != None):
            random.seed(self.seed)
            
        n_trees = []
        accuracies = []
        for i in range(self.N):
            header = []
            attribute_domains = {}
            
            #loops through X_train and creates header
            for i in range(len(X_train[0])) :
                header.append("att" + str(i))
            

            #loops though header to form attribute domains dictionairy
            count = 0
            for item in header:
                curr_col = myutils.get_column(X_train, count)
                values, counts = myutils.get_frequencies(curr_col)
                attribute_domains[item] = values
                count+=1
                

            #stitching together X_train and y_train and getting available attributes
            train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
            available_attributes = header.copy()

            boot_train = myutils.compute_bootstrapped_sample(train)

            validation_set = []
            for row in train:
                if row not in boot_train:
                    validation_set.append(row)



            #forming tree
            tree = myutils.tdidt_forest(boot_train, available_attributes, attribute_domains, header, self.F)
            #print(tree)

            tree_dict = {}
            tree_dict["tree"] = tree
            y_test = []
            for row in validation_set:
                y_test.append(row.pop())
            
            y_predict = myutils.predict_tree(validation_set, tree)

            acc = myutils.get_accuracy(y_predict, y_test)
            tree_dict["acc"] = acc
            n_trees.append(tree_dict)
        

        sorted_trees = sorted(n_trees, key=lambda k: k['acc'], reverse=True)
        for i in range(self.M):
            self.trees.append(sorted_trees[i]["tree"])
        

        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for item in X_test:
            tree_predicts = []
            for tree in self.trees:
                y_predict = myutils.predict_helper(item,tree)
                tree_predicts.append(y_predict[1])
            
            y_predicted.append(max(set(tree_predicts), key=tree_predicts.count))

        return y_predicted
