import numpy as np
import math
import csv
import random
import copy
#from tabulate import tabulate


def compute_euclidean_distance(v1, v2):
    """ computes euclidean distance for paralel lists passed in
    Args:
        vl: (list) list of values
        v2: (list) paralel list of values
    Returns:
        dist: euclidean distance of passed in lists
    """
    
    assert len(v1) == len(v2)

    dist = []
    if (isinstance(v1[0],str) or isinstance(v2[0],str)):
        for i in range(len(v1)):
            if (v1[i] == v2[i]):
                    dist.append(0)
            else:
                dist.append(1)
    else:
        dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
       
    return dist


def get_frequencies(col):
    """ gets frequencies for the passed in col_name and returns parallel arrays
    with the values in the collumns and the counts
    Args:
        col: (list) column name of frequencies to find
    Returns:
        values: (list) values in col_name
        counts: (list) paralel list to values list of frequency counts
    """

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts 


def get_column(table, col_index):
    """ gets the column from a passed in col_name
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        col: (list) column wanted
    """
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def get_column2(table, header, col_name):
    """ gets the column from a passed in col_name
    Args:
        table: (list of lists) table of data to get column from
        header: (list) header of table
        col_name: (list) column name of frequencies to find
    Returns:
        col: (list) column wanted
    """
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def group_by(table, col_index):
    """ groups the table by the passed in col_index
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        group_names: (list) list of group label names
        group_subtables: (list of lists) 2d list of each group subtable
    """
    col = get_column(table, col_index)

    # get a list of unique values for the column
    group_names = sorted(list(set(col))) # 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], []]

    # walk through each row and assign it to the appropriate
    # subtable based on its group by value (model year)
    for row in table:
        group_value = row[col_index]
        # which group_subtable??
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables

def create_data(csv_file):

    """ Loads the csv_file data into a 2D Python list and removes and stores the
    header
    Args:
        csv_file: csv file to remove header from
    Returns:
        table: csv file in table format
        header: header of new table
    """

    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        table = list(csv_reader)
        header = table.pop(0)
    return header,table

def convert_weight_to_categorical(table):
    """ converts wieght category to categorical
    Args:
        table: (list of lists) table of data
    """
    # runs through each item in col and assigns new categorical value from 1-10
    for i in range(len(table)):
        if (table[i][4] >= 3500):
            table[i][4] = 5
        elif (table[i][4] >=3000 and table[i][4] < 3500):
            table[i][4] = 4
        elif (table[i][4] >=2500 and table[i][4] < 3000):
            table[i][4] = 3
        elif (table[i][4] >=2000 and table[i][4] < 2500):
            table[i][4] = 2
        else:
            table[i][4] = 1

def convert_mpg_to_categorical(table):
    """ converts mpg category to categorical and returns the new collumn
    Args:
        table: (list of lists) table of data to get column from
        header: (list) header of table
        col_name: (list) column name of frequencies to find
    Returns:
        col: (list) new col of converted data
    """

    # runs through each item in col and assigns new categorical value from 1-10
    for i in range(len(table)):
        if (table[i][0] >= 45):
            table[i][0] = 10
        elif (table[i][0] >=37 and table[i][0] < 45):
            table[i][0] = 9
        elif (table[i][0] >=31 and table[i][0] < 37):
            table[i][0] = 8
        elif (table[i][0] >=27 and table[i][0] < 31):
            table[i][0] = 7
        elif (table[i][0] >=24 and table[i][0] < 27):
            table[i][0] = 6
        elif (table[i][0] >=20 and table[i][0] < 24):
            table[i][0] = 5
        elif (table[i][0] >=17 and table[i][0] < 20):
            table[i][0] = 4
        elif (table[i][0] >=15 and table[i][0] < 17):
            table[i][0] = 3
        elif (table[i][0] >= 14 and table[i][0] < 15):
            table[i][0] = 2
        else:
            table[i][0] = 1

def NaiveSetup(table):
    """ Sets up the Naive Classifiers X_train and y_train
    Args:
        table: (list of lists) table of data
    Returns:
        X_train: (list of list) X_train set for Niave Bayes Classifier
        y_train: (list) y_train set for Niave Bayes Classifier  
    """
    X_train =[]
    for i in range(len(table)):
        alist = []
        alist.append(table[i][1])
        alist.append(table[i][4])
        alist.append(table[i][6])
        X_train.append(alist)
    
    y_train = get_column(table,0)
    return X_train, y_train

def generate_titanic_dataset(table):
    """ Sets up the Titanic Classifiers X_train and y_train
    Args:
        table: (list of lists) table of data
    Returns:
        X_train: (list of list) X_train set for Titanic Set
        y_train: (list) y_train set for Titanic Set  
    """
    X_train = []
    y_train = []
    for i in range(len(table)):
        alist = table[i][0:3]
        X_train.append(alist)
        y_train.append(table[i][3])
    
    return X_train, y_train

def select_attribute(instances, available_attributes, header):

    """ Selects attributes using entropy
    Args:
        instances: (list of lists) table of data
        available_attributes: (list) list of available attributes to split on
        header: (list) header of attributes
    Returns:
        attribute: (string) attribute selected to split on
    """
    
    table_size = len(instances)
    e_new_list = []

    # loops through each available attribute and groups data by each attribute
    for item in available_attributes : 
        group_names, group_subtables = group_by(instances, header.index(item))
        e_value_list = []
        num_values = []

        # loops through the group subtable and further groups by class name
        for j in range(len(group_subtables)):
            curr_group = group_subtables[j]
            num_attributes = len(curr_group)
            num_values.append(num_attributes)
            class_names, class_subtables = group_by(curr_group, len(curr_group[0])-1)
            e_value = 0

            #checks for empty partition for log base 2 of 0 calculations
            if (len(class_subtables) == 1):
                    e_value = 0
            else :
                #loops through each group bay attribute class and calculates the entropy
                for k in range(len(class_subtables)):
                    class_num = len(class_subtables[k]) / num_attributes
                    e_value -= (class_num) * (math.log2(class_num))
            e_value_list.append(e_value)
        
        e_new = 0

        #calculates e_new for each attribute 
        for l in range (len(e_value_list)):
            e_new += e_value_list[l] * (num_values[l]/ table_size)
        e_new_list.append(e_new)

    #finds attribute with minimum entropy and selects that attribute
    min_entropy = min(e_new_list)
    min_index = e_new_list.index(min_entropy)
    attribute = available_attributes[min_index]

    return attribute

def partition_instances(instances, split_attribute, attribute_domains, header):
    """ partitions the passed in instances by split instance
    Args:
        instances: (list of lists) table of data
        split_attribute: (string) attribute to split on
        attribute_domains: (dict) domains for the attributes
        header: (list) header of attributes
    Returns:
        partitions: (dict) new partitions by split attribute
    """
    # comments refer to split_attribute "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0

    partitions = {} # key (attribute value): value (partition)
    # task: try to finish this
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions 

def all_same_class(instances):
    """ checks if all of the instances in instances have the same class and returns true if so
    Args:
        instances: (list of lists) table of data
    Returns:
        true if instances are all same class, false otherwise
    """
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def calc_majority_leaf(partition):
    """ calculates the majority class leaf node of the partition and returns the classification
    Args:
        partition: (list of lists) partitions of data
    Returns:
        classification: (String) majority vote classification of partitions
    """
    col = []
    classification = ""
    for item in partition:
        col.append(item[-1])

    values, counts = get_frequencies(col)
    max_num = max(counts)
    max_index = counts.index(max_num)
    classification = values[max_index]

    return classification


def tdidt_forest(current_instances, available_attributes, attribute_domains, header,F):
    """ Recursive helper function to help form the tree
    Args:
        current_instances: (list of lists) table of data
        available_attributes: (list) available attributes for splitting
        attribute_domains: (dict) domains for the attributes
        header: (list) header of attributes
    Returns:
        tree: (nested list) decision tree created as a nested list
    """



    atts = compute_random_subset(available_attributes, F)
    
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, atts, header)

    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch

    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    atts.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)

    # for each partition, repeat unless one of the following occurs (base case)
    Skip = False
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            leaf_node = ["Leaf", partition[0][-1]]
            values_subtree.append(leaf_node)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            classification = calc_majority_leaf(partition)
            leaf_node = ["Leaf", classification]
            values_subtree.append(leaf_node)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:

            values = []
            #loops trhough each current partition and further each item in the partitions
            for attribute_value, partition in partitions.items():
                for item in partition:

                    #checks if the partition isn't empty and adds them to a list
                    if len(item) != 0:
                        values.append(item)

            #calculates the majority leaf node of the values 
            classification = calc_majority_leaf(values)

            #sets the current attribute to a leaf node
            tree = ["Leaf", classification]
            Skip = True
        else: # all base cases are false, recurse!!
            subtree = tdidt_forest(partition, available_attributes.copy(), attribute_domains, header,F)
            values_subtree.append(subtree)
        #if case 3 didn't occur, the tree appends the values subtree
        if (Skip == False):
            tree.append(values_subtree)
    return tree


def tdidt(current_instances, available_attributes, attribute_domains, header):
    """ Recursive helper function to help form the tree
    Args:
        current_instances: (list of lists) table of data
        available_attributes: (list) available attributes for splitting
        attribute_domains: (dict) domains for the attributes
        header: (list) header of attributes
    Returns:
        tree: (nested list) decision tree created as a nested list
    """



    
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes, header)

    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch

    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)

    # for each partition, repeat unless one of the following occurs (base case)
    Skip = False
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            leaf_node = ["Leaf", partition[0][-1]]
            values_subtree.append(leaf_node)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            classification = calc_majority_leaf(partition)
            leaf_node = ["Leaf", classification]
            values_subtree.append(leaf_node)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:

            values = []
            #loops trhough each current partition and further each item in the partitions
            for attribute_value, partition in partitions.items():
                for item in partition:

                    #checks if the partition isn't empty and adds them to a list
                    if len(item) != 0:
                        values.append(item)

            #calculates the majority leaf node of the values 
            classification = calc_majority_leaf(values)

            #sets the current attribute to a leaf node
            tree = ["Leaf", classification]
            Skip = True
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            values_subtree.append(subtree)
        #if case 3 didn't occur, the tree appends the values subtree
        if (Skip == False):
            tree.append(values_subtree)
    return tree

def predict_helper(X_test, curr_tree) :
    """ Recursive helper function for predicting a class in the decision tree
    Args:
        X_test: (list) current X_test being predicted
        curr_tree: (nested list) current tree being passed in recursively
    Returns:
        curr_tree: (list) final leaf node found for classification
    """

    if (curr_tree[0] == "Attribute"):
       
        #getting attribute value from X_test
        curr_string = curr_tree[1]
        curr_index = int(curr_string[3])
    
        curr_value = X_test[curr_index]

        for i in range(2, len(curr_tree)):

            #check to see if curr_value = the current partion in the tree then recursively returns the new tree
            if curr_value == curr_tree[i][1]:
                curr_tree = curr_tree[i]
                return predict_helper(X_test,curr_tree)
                

    #checks for end case, leaf is found
    elif ("Leaf" in curr_tree):
        return curr_tree
    
    # returns new tree if value is the lead of the current subtree
    curr_tree = curr_tree[2]
    return predict_helper(X_test,curr_tree)

def print_tree_helper(tree, rule, curr_att, attribute_names=None, class_name="class"):
    """ Recursive helper function for printing the rules of a tree
    Args:
        tree(nested list): current subtree being passed recursively
        rule(list): current rule being formed
        curr_att(string): current attribute of tree to keep track of rules
        attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
        class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        
    Returns:
        tree[1]: (string) final leaf node in tree, used to end function
    """

    info_type = tree[0]

    #checks if recursion needed
    if info_type == "Attribute":

        #gets current attribute and appends it to tree
        curr_att = tree[1]
        rule.append(tree[1])

        #loops trhough all values in current subtree
        for i in range(2, len(tree)):

            #reforms current rule based on the current attribute
            value_list = tree[i]
            curr_index = len(rule) - 1
            att_index = rule.index(curr_att)

            #deletets item from current rule index of current attribute is found
            while (curr_index != att_index):
                del rule[-1]
                curr_index -= 1

            # appends new value to rule
            rule.append("==")
            rule.append(value_list[1])
            rule.append("and")
            
            print_tree_helper(value_list[2], rule, curr_att, attribute_names, class_name)

    # leaf is found
    else: 

        # Prints out a rule
        print("If", end=" ")
        del rule[-1]
        for item in rule:
            if isinstance(item,str):
                if ("att" in item):
                    if (attribute_names != None):
                        print(attribute_names[int(item[3])], end= " ")
                    else:
                        print(item, end=" ")
                else:
                    print(item, end=" ")
            else:
                print(item, end=" ")
        
        print(", Then", class_name, "=", tree[1])
        print()

        #returns last leaf to end function
        return tree[1]

def predict_tree(X_test, tree):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        y_predict = []
        for item in X_test:
            y_predict = predict_helper(item,tree)
            y_predicted.append(y_predict[1])

        return y_predicted

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

def compute_bootstrapped_sample(table, seed_num=None):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample 

def compute_random_subset(values, num_values):
    shuffled = values[:]
    random.shuffle(shuffled)
    return shuffled[:num_values]


    



    


    


