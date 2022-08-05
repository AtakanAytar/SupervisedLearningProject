#%%
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
#%%

# # Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        rf = joblib.load("randomforest.pkl")
        

        ###we have to get the values from the front end 

        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        prediction = rf.predict(X)[0]
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)
#%%
#I think we should either have a route for each of these models or send something to Main fn
@app.route("/nn", methods = ["GET","POST"])
def get_nn():
    if request.method == "Neural Network Model":
        # Unpickle classifier
        rf = joblib.load("NeuralNetwork.pkl")

        ###we have to get the values from the front end 

        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        prediction = rf.predict(X)[0]
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)

@app.route("/tree", methods = ["GET","POST"])
def get_nn():
    if request.method == "Decision Tree Model":
        # Unpickle classifier
        rf = joblib.load("DecTree.pkl")

        ###we have to get the values from the front end 

        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        prediction = rf.predict(X)[0]
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)

@app.route("/logreg", methods = ["GET","POST"])
def get_nn():
    if request.method == "LogisticRegression Model":
        # Unpickle classifier
        rf = joblib.load("LogReg.pkl")

        ###we have to get the values from the front end 

        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        prediction = rf.predict(X)[0]
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)

@app.route("/svm", methods = ["GET","POST"])
def get_nn():
    if request.method == "SVM Model":
        # Unpickle classifier
        rf = joblib.load("SVM.pkl")

        ###we have to get the values from the front end 

        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        prediction = rf.predict(X)[0]
        
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)