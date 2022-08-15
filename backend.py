
from flask import Flask, request, render_template, jsonify
import pandas as pd
import json
import joblib


# # Declare a Flask app
app = Flask(__name__)

model_pkl_map = {
    'logistic': ['./models/LogReg.pkl', "./reports/logreg_results.json"],
    'forest': ['./models/randomforest.pkl', "./reports/forest_results.json"], 
    'nueral': ['./models/NeuralNetwork.pkl', "./reports/nnetwork_results.json"],
    'decision': ['./models/DecTree.pkl', "./reports/dectree_results.json"],
    'svm': ['./models/SVM.pkl', "./reports/fsvm_results.json"]
}


@app.route('/api', methods=['GET', 'POST'])
def api_route():
    response = {}

    if request.method == "POST":
        # get/validate input
        input = request.get_json()
        if not input['model'] or not input['data']:
            raise Exception("Invalid request")

        # extract the model name and the test data
        # and covert into dataframe
        model_name = input['model']
        test_data = input['data']

        test_data_dict = {}
        for key in test_data:
            test_data_dict[key] = pd.Series([test_data[key]])
        X = pd.DataFrame(test_data_dict)

        # load/deserialize the model from the pkl file
        if model_name not in model_pkl_map:
            raise Exception("Unknown model name: " + model_name)

        plk_filename = model_pkl_map[model_name][0]
        classifier = joblib.load(plk_filename)
        if not classifier:
            raise Exception("Unable to load model with model name: " + model_name)

        # predict using the model
        prediction = classifier.predict(X)[0]
        # load the score for the model
        score = json.load(open(model_pkl_map[model_name][1],"r"))
        
        response['prediction'] = prediction
        response['score'] = score['weighted avg']
        response['score']['accuracy'] = score['accuracy']
    else:
        response['message'] = "Please send POST request to run the model"

    # return the result back to client
    return jsonify(response)


@app.route('/', methods=['GET', 'POST'])
def main():
    prediction = {}
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        rf = joblib.load("./models/randomforest.pkl")
        ##we have to get the values from the front end 
        input = request.get_json()
        x_dict = {}
        for ikey in input:
            x_dict[ikey] = pd.Series([input[ikey]])
        X = pd.DataFrame(x_dict)
        # Get prediction
        prediction = rf.predict(X)[0]
        #prediction = rf.score(X)[0]
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)

@app.route('/forest', methods=['GET', 'POST'])
def get_forest():
    prediction = {}~
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        rf = joblib.load("./models/randomforest.pkl")
       ###we have to get the values from the front end 
        input = request.get_json()
        x_dict = {}
        for ikey in input:
            x_dict[ikey] = pd.Series([input[ikey]])
        X = pd.DataFrame(x_dict)
        # Get prediction
        prediction = rf.predict(X)[0]
        score = json.load(open("./reports/forest_results.json","r"))
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction,score)

#I think we should either have a route for each of these models or send something to Main fn
@app.route("/nn", methods = ["GET","POST"])
def get_nn():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("./models/NeuralNetwork.pkl")
        ###we have to get the values from the front end 
        input = request.get_json()
        x_dict = {}
        for ikey in input:
            x_dict[ikey] = pd.Series([input[ikey]])
        X = pd.DataFrame(x_dict)
        # Get prediction
        prediction = rf.predict(X)[0]
        score = json.load(open("./reports/nnetwork_results.json","r"))
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction,score)

@app.route("/tree", methods = ["GET","POST"])
def get_tree():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("./models/DecTree.pkl")
        ##we have to get the values from the front end 
        input = request.get_json()
        x_dict = {}
        for ikey in input:
            x_dict[ikey] = pd.Series([input[ikey]])
        X = pd.DataFrame(x_dict)
        # Get prediction
        prediction = rf.predict(X)[0]
        score = json.load(open("./reports/dectree_results.json","r"))
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction,score)

@app.route("/logreg", methods = ["GET","POST"])
def get_log():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("./models/LogReg.pkl")
        ##we have to get the values from the front end 
        input = request.get_json()
        x_dict = {}
        for ikey in input:
            x_dict[ikey] = pd.Series([input[ikey]])
        X = pd.DataFrame(x_dict)
        # Get prediction
        prediction = rf.predict(X)[0]
        score = json.load(open("./reports/logreg_results.json","r"))
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction,score)

@app.route("/svm", methods = ["GET","POST"])
def get_svm():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("./models/SVM.pkl")
        ##we have to get the values from the front end 
        input = request.get_json()
        x_dict = {}
        for ikey in input:
            x_dict[ikey] = pd.Series([input[ikey]])
        X = pd.DataFrame(x_dict)
        # Get prediction
        prediction = rf.predict(X)[0]
        score = json.load(open("./reports/fsvm_results.json","r"))
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction,score)

@app.route("/test", methods = ["GET","POST"])
def test():
    if request.method == "POST":
        #print(jsonify(request.data))
        return request.get_json()

# Running the app
if __name__ == '__main__':
    app.run(debug = True, use_reloader = False)