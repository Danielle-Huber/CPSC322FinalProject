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

    wife_age = float(wife_age)
    wife_education = float(wife_education)
    number_of_children = float(number_of_children)
    husband_occupation = float(husband_occupation)

    # 0 1 3 6
    # wife's age, wife's education, number of children, husband's occupation

    instance = [wife_age, wife_education, number_of_children, husband_occupation]
    print(instance)
    prediction = predict_contraceptive(instance)
    if prediction is not None:
        result = {"prediction" : prediction} 
        return jsonify(result), 200
    else:
        return "Error making prediction", 400


def predict_contraceptive(instance):
    infile = open("bayes.p", "rb")
    bayes = pickle.load(infile)
    infile.close()

    try:
        predictions = bayes.predict([instance])
        print(predictions)
        return predictions[0]
    except:
        return None



if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=port)


