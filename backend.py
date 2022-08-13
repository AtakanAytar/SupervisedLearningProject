
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib


# # Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        rf = joblib.load("randomforest.pkl")
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
    prediction = {}
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        rf = joblib.load("randomforest.pkl")
       ###we have to get the values from the front end 
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

#I think we should either have a route for each of these models or send something to Main fn
@app.route("/nn", methods = ["GET","POST"])
def get_nn():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("./NeuralNetwork.pkl")
        ###we have to get the values from the front end 
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

@app.route("/tree", methods = ["GET","POST"])
def get_tree():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("DecTree.pkl")
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

@app.route("/logreg", methods = ["GET","POST"])
def get_log():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("LogReg.pkl")
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

@app.route("/svm", methods = ["GET","POST"])
def get_svm():
    prediction = {}
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("SVM.pkl")
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

@app.route("/test", methods = ["GET","POST"])
def test():
    if request.method == "POST":
        #print(jsonify(request.data))
        return request.get_json()

# Running the app
if __name__ == '__main__':
    app.run(debug = True, use_reloader = False)