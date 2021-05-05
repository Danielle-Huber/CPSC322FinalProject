from mysklearn.myclassifiers import MyRandomForestClassifier,  MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import numpy as np

def test_random_forest_fit():
    X = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    test_trees = [
        ['Attribute', 'att0', 
            ['Value', 'Junior', 
                ['Attribute', 'att2', 
                    ['Value', 'no', 
                        ['Leaf', 'False']
                    ], 
                    ['Value', 'yes', 
                        ['Attribute', 'att3', 
                            ['Value', 'no', 
                                ['Leaf', 'True']
                            ], 
                            ['Value', 'yes', 
                                ['Leaf', 'False']
                            ]
                        ]
                    ]
                ]
            ], 
            ['Value', 'Mid', 
                ['Leaf', 'True']
            ], 
            ['Value', 'Senior', 
                ['Attribute', 'att2', 
                    ['Value', 'no', 
                        ['Leaf', 'False']
                    ], 
                    ['Value', 'yes', 
                        ['Leaf', 'True']
                    ]
                ]
            ]
        ],

        ['Attribute', 'att0', 
            ['Value', 'Junior', 
                ['Leaf', 'False']
            ], 
            ['Value', 'Mid', 
                ['Leaf', 'True']
            ], 
            ['Value', 'Senior', 
                ['Attribute', 'att2', 
                    ['Value', 'no', 
                        ['Leaf', 'False']
                    ], 
                    ['Value', 'yes',  
                        ['Leaf', 'True']
                    ]
                ]
            ]
        ]
    ]
    forest = MyRandomForestClassifier(n=4, m=2,f=2, seed = 2)
    forest.fit(X,y)

    assert forest.trees == test_trees
    #print(forest.trees)

def test_random_forest_predict():
    X = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    forest = MyRandomForestClassifier(n=4, m=2,f=2, seed = 2)
    forest.fit(X,y)
    y_predicted = forest.predict([["Junior", "Python", "no", "yes"],["Mid", "Java","yes","no"]])
    y_actual = ['False', 'True']
    assert y_predicted == y_actual

def test_kneighbors_classifier_kneighbors():
    KNC = MyKNeighborsClassifier()

    X_train = [
        [1, 1],
        [1, 0],
        [.33, 0],
        [0, 0],
    ]
    y_train = ["bad", "bad", "good", "good"]
    KNC.fit(X_train, y_train)
    distances, indices = KNC.kneighbors([[.33,1]])
    test_distance = [[0.67,1,1.05304]]
    test_indices = [[0,2,3]]

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            print(distances[i][j])
            assert np.isclose(distances[i][j],test_distance[i][j])

    for i in range (len(indices)):
        for j in range(len(indices[i])):
            assert np.isclose(indices[i][j], test_indices[i][j])

    ### Test #2 ###
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    KNC.fit(X_train, y_train)
    distances, indices = KNC.kneighbors([[2,3]])
    test_distance = [[1.4142135623730951,1.4142135623730951,2.0]]
    test_indices = [[0,4,6]]

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            assert np.isclose(distances[i][j],test_distance[i][j])

    for i in range (len(indices)):
        for j in range(len(indices[i])):
            assert np.isclose(indices[i][j], test_indices[i][j])


    ### Test #3 ###
    KNC = MyKNeighborsClassifier(n_neighbors=5)

    X_train = [
        [.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [17.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]
    ]
    y_train = ["-","-","-","+","-","+","-","+","+","+","-","-","-","-",
    "-","+","+","+","-","+"]
    KNC.fit(X_train, y_train)
    distances, indices = KNC.kneighbors([[9.1,11]])
    print(distances)
    print(indices)
    
    test_distance = [[0.6082762530298216, 1.2369316876852974, 2.202271554554525, 2.8017851452243794, 2.9154759474226513]]
    test_indices = [[6, 5, 7, 4, 8]]

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            assert np.isclose(distances[i][j],test_distance[i][j])

    for i in range (len(indices)):
        for j in range(len(indices[i])):
            assert np.isclose(indices[i][j], test_indices[i][j])


def test_kneighbors_classifier_predict():

    ### Test #1 ###
    KNC = MyKNeighborsClassifier()

    X_train = [
        [1, 1],
        [1, 0],
        [.33, 0],
        [0, 0],
    ]
    y_train = ["bad", "bad", "good", "good"]
    KNC.fit(X_train, y_train)
    y_predicted = KNC.predict([[0.33,1]])
    test_predicted = ["good"]

    for i in range(len(y_predicted)):
        assert y_predicted[i] == test_predicted[i]
    
    ### Test #2 ###
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    KNC.fit(X_train, y_train)
    y_predicted = KNC.predict([[2,3]])
    test_predicted = ["yes"]

    for i in range(len(y_predicted)):
        assert y_predicted[i] == test_predicted[i]


    ### Test #3 ###
    KNC = MyKNeighborsClassifier(n_neighbors=5)

    X_train = [
        [.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [17.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]
    ]
    y_train = ["-","-","-","+","-","+","-","+","+","+","-","-","-","-",
    "-","+","+","+","-","+"]
    KNC.fit(X_train, y_train)
    y_predicted = KNC.predict([[9.1,11]])
    test_predicted = ["+"]

    for i in range(len(y_predicted)):
        assert y_predicted[i] == test_predicted[i]

def test_naive_bayes_classifier_fit():

    ###Test #1###
    col_names = ["att1", "att2"]
    X_train = [
         [1,5],
         [2,6],
         [1,5],
         [1,5],
         [1,6],
         [2,6],
         [1,5],
         [1,6]
    ]
    y_train = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    Naive = MyNaiveBayesClassifier()
    Naive.fit(X_train,y_train)

    prior_ans = [5/8,3/8]
    post_ans = [
        [
            ["Attributes", "yes", "no"],
            [1, 4/5, 2/3],
            [2, 1/5, 1/3]
        ],

        [
            ["Attributes", "yes", "no"],
            [5, 2/5, 2/3],
            [6, 3/5, 1/3]
        ]
    ]

    for i in range(len(post_ans)):
        for j in range(len(post_ans[i])):
            for k in range(len(post_ans[i][j])):
                if (isinstance(post_ans[i][j][k],float)):
                    post_ans[i][j][k] = round(post_ans[i][j][k],3)
    
    for i in range(len(prior_ans)):
        assert np.isclose(Naive.priors[i], prior_ans[i])

    for i in range(len(post_ans)):
        for j in range(len(post_ans[i])):
            for k in range (len(post_ans[i][j])):
                n = Naive.posteriors[i][j][k]
                ans = post_ans[i][j][k]
                assert n == ans

    ###Test #2###

    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train = [
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [1, 3, "fair"],
        [1, 3, "excellent"]
    ]
    y_train = ["yes","yes","yes","no","yes","no","yes","yes","yes","yes","yes","no","yes","no","no"]
    

    Naive = MyNaiveBayesClassifier()
    Naive.fit(X_train,y_train)

    prior_ans = [10/15,5/15]
    post_ans = [
        [
            ["Attributes","yes","no"],
            [1,2/10,3/5],
            [2,8/10,2/5]
        ],

        [
            ["Attributes","yes","no"],
            [1,3/10,1/5],
            [2,4/10,2/5],
            [3,3/10,2/5]
        ],

        [
            ["Attributes","yes","no"],
            ["fair",7/10,2/5],
            ["excellent",3/10,3/5]
        ]
    ]
    for i in range(len(post_ans)):
        for j in range(len(post_ans[i])):
            for k in range(len(post_ans[i][j])):
                if (isinstance(post_ans[i][j][k],float)):
                    round(post_ans[i][j][k],3)

    for i in range(len(prior_ans)):
        assert prior_ans[i] in Naive.priors

    for i in range(len(post_ans)):
        for j in range(len(post_ans[i])):
            assert post_ans[i][j] in Naive.posteriors[i]
    
    ###Test #3###
    
    train_col_names = ["day", "season", "wind", "rain", "class"]
    X_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"], 
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]

    y_train = ["on time","on time","on time","late","on time","very late","on time","on time","very late",
        "on time","cancelled","on time","late","on time","very late", "on time","on time","on time","on time",
        "on time"]
    

    Naive = MyNaiveBayesClassifier()
    Naive.fit(X_train,y_train)
    
    prior_ans = [0.7, 0.1, 0.15, 0.05]
    post_ans = [
        [
            ['Attributes', 'on time', 'late', 'very late', 'cancelled'], 
            ['weekday', 0.643, 0.5, 1.0, 0.0], 
            ['saturday', 0.143, 0.5, 0.0, 1.0], 
            ['holiday', 0.143, 0.0, 0.0, 0.0], 
            ['sunday', 0.071, 0.0, 0.0, 0.0]
            ], 
            
        [   
            ['Attributes', 'on time', 'late', 'very late', 'cancelled'], 
            ['spring', 0.286, 0.0, 0.0, 1.0],
            ['winter', 0.143, 1.0, 0.667, 0.0], 
            ['summer', 0.429, 0.0, 0.0, 0.0], 
            ['autumn', 0.143, 0.0, 0.333, 0.0]
            ], 
            
        [   ['Attributes', 'on time', 'late', 'very late', 'cancelled'], 
            ['none', 0.357, 0.0, 0.0, 0.0], 
            ['high', 0.286, 0.5, 0.333, 1.0], 
            ['normal', 0.357, 0.5, 0.667, 0.0]
            ], 
        [
            ['Attributes', 'on time', 'late', 'very late', 'cancelled'], 
            ['none', 0.357, 0.5, 0.333, 0.0], 
            ['slight', 0.571, 0.0, 0.0, 0.0], 
            ['heavy', 0.071, 0.5, 0.667, 1.0]
        ]
    
    ]
    prior_ans = [0.7, 0.1, 0.15, 0.05]

    
    for i in range(len(post_ans)):
        for j in range(len(post_ans[i])):
            for k in range(len(post_ans[i][j])):
                if (isinstance(post_ans[i][j][k],float)):
                    round(post_ans[i][j][k],3)

    for i in range(len(prior_ans)):
        assert prior_ans[i] in Naive.priors

    for i in range(len(post_ans)):
        for j in range(len(post_ans[i])):
            assert post_ans[i][j] in Naive.posteriors[i]
    
def test_naive_bayes_classifier_predict():

    ###Test #1###
    col_names = ["att1", "att2"]
    X_train = [
         [1,5],
         [2,6],
         [1,5],
         [1,5],
         [1,6],
         [2,6],
         [1,5],
         [1,6]
    ]
    y_train = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    Naive = MyNaiveBayesClassifier()
    Naive.fit(X_train,y_train)
    y_predict = Naive.predict([[1,5]])
    y_actual = ["yes"]

    assert y_predict[0] == y_actual[0]

    ###Test #2###

    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train = [
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [1, 3, "fair"],
        [1, 3, "excellent"]
    ]
    y_train = ["yes","yes","yes","no","yes","no","yes","yes","yes","yes","yes","no","yes","no","no"]
    

    Naive = MyNaiveBayesClassifier()
    Naive.fit(X_train,y_train)

    y_predict = Naive.predict([[2, 2, "fair"]])
    y_actual = ["yes"]
    assert y_predict[0] == y_actual[0]

    y_predict = Naive.predict([[1, 1, "excellent"]])
    y_actual = ["no"]
    assert y_predict[0] == y_actual[0]

    ###Test #3###

    train_col_names = ["day", "season", "wind", "rain", "class"]
    X_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"], 
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]

    y_train = ["on time","on time","on time","late","on time","very late","on time","on time","very late",
        "on time","cancelled","on time","late","on time","very late", "on time","on time","on time","on time",
        "on time"]
    

    Naive = MyNaiveBayesClassifier()
    Naive.fit(X_train,y_train)

    y_predict = Naive.predict([["weekday","winter","high","heavy"],["saturday","spring","normal","slight"]])
    y_actual = ["very late","on time"]
    for i in range(len(y_actual)):
        assert y_predict[i] == y_actual[i]




def test_decision_tree_classifier_fit():
    ### TEST #1 ###

    interview_tree = \
 ['Attribute', 'att0', 
    ['Value', 'Junior', 
        ['Attribute', 'att3', 
            ['Value', 'no', 
                ['Leaf', 'True']
            ], 
            ['Value', 'yes', 
                ['Leaf', 'False']
            ]
        ]
    ], 
    ['Value', 'Mid', 
        ['Leaf', 'True']
    ], 
    ['Value', 'Senior', 
        ['Attribute', 'att2', 
            ['Value', 'no', 
                ['Leaf', 'False']
            ], 
            ['Value', 'yes', 
                ['Leaf', 'True']
            ]
        ]
    ]
    ]

    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    tree_classifier = MyDecisionTreeClassifier()
    tree_classifier.fit(X_train,y_train)
    assert  tree_classifier.tree == interview_tree

    ### Test #2 ###
    degree_tree =  \
    ['Attribute', 'att0', 
        ['Value', 'A', 
            ['Attribute', 'att4', 
                ['Value', 'A', 
                    ['Leaf', 'First']
                ], 
                ['Value', 'B', 
                    ['Attribute', 'att3', 
                        ['Value', 'A', 
                            ['Attribute', 'att1', 
                                ['Value', 'A', 
                                    ['Leaf', 'First']
                                ], 
                                ['Value', 'B', 
                                    ['Leaf', 'Second']
                                ]
                            ]
                        ], 
                        ['Value', 'B', 
                            ['Leaf', 'Second']
                        ]
                    ]
                ]
            ]
        ], 
        ['Value', 'B', 
            ['Leaf', 'Second']
        ]
    ]

    X_train = [
    ["A", "B", "A", "B" ,"B"],
    ["A", "B", "B", "B", "A"], 
    ["A", "A", "A", "B", "B"], 
    ["B", "A", "A","B", "B"], 
    ["A", "A", "B", "B", "A"], 
    ["B", "A", "A", "B", "B"] ,
    ["A", "B", "B", "B", "B"], 
    ["A", "B", "B", "B", "B"], 
    ["A", "A", "A", "A", "A"], 
    ["B", "A", "A", "B", "B"], 
    ["B", "A", "A", "B", "B"], 
    ["A", "B", "B", "A", "B"], 
    ["B", "B", "B", "B", "A"], 
    ["A", "A", "B", "A", "B"],
    ["B", "B", "B", "B", "A"], 
    ["A", "A", "B", "B", "B"], 
    ["B", "B", "B", "B", "B"], 
    ["A", "A", "B", "A", "A"], 
    ["B", "B", "B", "A", "A"], 
    ["B", "B", "A", "A", "B"], 
    ["B", "B", "B", "B", "A"], 
    ["B", "A", "B", "A", "B"], 
    ["A", "B", "B", "B", "A"], 
    ["A", "B", "A", "B", "B"], 
    ["B", "A", "B", "B", "B"], 
    ["A", "B", "B", "B", "B"]
    ] 

    y_train = ["Second", "First", "Second", "Second", "First", "Second",
    "Second","Second", "First", "Second", "Second", "Second", "Second", "First", "Second", "Second", "Second", "First",
    "Second", "Second", "Second", "Second", "First", "Second", "Second", "Second"]
    tree_classifier2 = MyDecisionTreeClassifier()
    tree_classifier2.fit(X_train,y_train)
    assert  tree_classifier2.tree == degree_tree
    
def test_decision_tree_classifier_predict():
    ## Test #1 ##
    interview_tree = \
 ['Attribute', 'att0', 
    ['Value', 'Junior', 
        ['Attribute', 'att3', 
            ['Value', 'no', 
                ['Leaf', 'True']
            ], 
            ['Value', 'yes', 
                ['Leaf', 'False']
            ]
        ]
    ], 
    ['Value', 'Mid', 
        ['Leaf', 'True']
    ], 
    ['Value', 'Senior', 
        ['Attribute', 'att2', 
            ['Value', 'no', 
                ['Leaf', 'False']
            ], 
            ['Value', 'yes', 
                ['Leaf', 'True']
            ]
        ]
    ]
    ]

    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    tree_classifier = MyDecisionTreeClassifier()
    tree_classifier.fit(X_train,y_train) 

    prediction = tree_classifier.predict([["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]])
    y_actual = ["True", "False"]
    assert prediction == y_actual

    ### Test #2 ###
    degree_tree =  \
    ['Attribute', 'att0', 
        ['Value', 'A', 
            ['Attribute', 'att4', 
                ['Value', 'A', 
                    ['Leaf', 'First']
                ], 
                ['Value', 'B', 
                    ['Attribute', 'att3', 
                        ['Value', 'A', 
                            ['Attribute', 'att1', 
                                ['Value', 'A', 
                                    ['Leaf', 'First']
                                ], 
                                ['Value', 'B', 
                                    ['Leaf', 'Second']
                                ]
                            ]
                        ], 
                        ['Value', 'B', 
                            ['Leaf', 'Second']
                        ]
                    ]
                ]
            ]
        ], 
        ['Value', 'B', 
            ['Leaf', 'Second']
        ]
    ]

    X_train = [
    ["A", "B", "A", "B" ,"B"],
    ["A", "B", "B", "B", "A"], 
    ["A", "A", "A", "B", "B"], 
    ["B", "A", "A","B", "B"], 
    ["A", "A", "B", "B", "A"], 
    ["B", "A", "A", "B", "B"] ,
    ["A", "B", "B", "B", "B"], 
    ["A", "B", "B", "B", "B"], 
    ["A", "A", "A", "A", "A"], 
    ["B", "A", "A", "B", "B"], 
    ["B", "A", "A", "B", "B"], 
    ["A", "B", "B", "A", "B"], 
    ["B", "B", "B", "B", "A"], 
    ["A", "A", "B", "A", "B"],
    ["B", "B", "B", "B", "A"], 
    ["A", "A", "B", "B", "B"], 
    ["B", "B", "B", "B", "B"], 
    ["A", "A", "B", "A", "A"], 
    ["B", "B", "B", "A", "A"], 
    ["B", "B", "A", "A", "B"], 
    ["B", "B", "B", "B", "A"], 
    ["B", "A", "B", "A", "B"], 
    ["A", "B", "B", "B", "A"], 
    ["A", "B", "A", "B", "B"], 
    ["B", "A", "B", "B", "B"], 
    ["A", "B", "B", "B", "B"]
    ] 

    y_train = ["Second", "First", "Second", "Second", "First", "Second",
    "Second","Second", "First", "Second", "Second", "Second", "Second", "First", "Second", "Second", "Second", "First",
    "Second", "Second", "Second", "Second", "First", "Second", "Second", "Second"]
    tree_classifier2 = MyDecisionTreeClassifier()
    tree_classifier2.fit(X_train,y_train)

    prediction = tree_classifier2.predict([["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]])
    y_actual = ["Second", "First", "First"]
    assert prediction == y_actual
