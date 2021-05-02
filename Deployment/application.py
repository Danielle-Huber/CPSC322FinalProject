from flask import Flask, jsonify, request
import os
import pickle
import mysklearn.myutils as myutils


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "Welcome to my App!", 200 

@app.route("/predict", methods=["GET"])
def predict():
    wife_age = request.args.get("wife_age", "")
    wife_education = request.args.get("wife_education", "")
    number_of_children = request.args.get("number_of_children", "")
    husband_occupation = request.args.get("husband_occupation", "")

    # 0 1 3 6
    # wife's age, wife's education, number of children, husband's occupation

    prediction = predict_contraceptive([wife_age, wife_education, number_of_children, husband_occupation])
    if prediction is not None:
        result = {"prediction" : prediction} 
        return jsonify(result), 200
    else:
        return "Error making prediction", 400

def bayes_predict(priors, posteriors, instance):
    X_test = [instance]
    probabilities = []
    y_predicted = []
    for i in range(len(X_test)):
        probabilities = []
        for m in range(len(priors)):
            prob = 1
            for k in range(len(posteriors)):
                currMatrix = posteriors[k]
                currCol = myutils.get_column(currMatrix,m+1)
                for j in range(len(currMatrix)):
                    if (currMatrix[j][0] == X_test[i][k]):
                        prob = prob * currCol[j]
            probabilities.append(prob * priors[m])            
        maxProb = probabilities.index(max(probabilities))
        y_predicted.append(posteriors[0][0][maxProb +1])

    return y_predicted


def predict_contraceptive(instance):
    infile = open("probabilities.p", "rb")
    priors, posteriors = pickle.load(infile)
    infile.close()

    try:
        return bayes_predict(priors, posteriors, instance)
    except:
        return None



if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=port)


