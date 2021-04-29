from mysklearn.myclassifiers import MyRandomForestClassifier

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
