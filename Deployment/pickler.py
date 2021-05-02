import pickle 

# for your project, pickle an instance MyRandomClassifier, MyDecisionTreeClassifer
# ask Gina this question ^^

packaged_object = [priors, posteriors]
outfile = open("probabilities.p", "wb")
pickle.dump(packaged_object, outfile)

outfile.close()