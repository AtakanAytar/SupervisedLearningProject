
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

@app.route("/test", methods = ["GET","POST"])
def test():
    if request.method == "POST":
        #print(jsonify(request.data))
        return request.get_json()
        

#I think we should either have a route for each of these models or send something to Main fn
@app.route("/nn", methods = ["GET","POST"])
def get_nn():
    if request.method == "POST":
        # Unpickle classifier
        rf = joblib.load("./NeuralNetwork.pkl")
        req_json = request.get_json()
        #req_json = jsonify(req_json)
        
        ###we have to get the values from the front end 
        
        test_json = {"HOUR":20,"TIME":2038,"STREET1":"FINCH Ave W","STREET2":"DUFFERIN St","DISTRICT":"North York","HOOD_ID":27,"TRAFFCTL":"No Control","VISIBILITY":"Rain","LIGHT":"Dark","RDSFCOND":"Wet","IMPACTYPE":"Pedestrian Collisions","INVTYPE":"Driver","INVAGE":"15 to 19","VEHTYPE":"Automobile, Station Wagon","LONGITUDE":-79.46989,"LATITUDE":43.768145}
        #json to dataframe
        columnss = ['HOUR', 'TIME', 'STREET1', 'STREET2', 'DISTRICT', 'HOOD_ID', 'TRAFFCTL',
       'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INVAGE',
       'VEHTYPE', 'LONGITUDE', 'LATITUDE']
        
        #test_json = jsonify(test_json)
        X= pd.DataFrame.from_dict(req_json)
        #X = pd.DataFrame.read_json(req_json)
        print(X)
        # Get prediction
        prediction = rf.predict(X)[0]
        print(type(req_json))
    else:
        prediction = ""
    ###return a json here   
    return jsonify(prediction)

@app.route("/tree", methods = ["GET","POST"])
def get_tree():
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
def get_log():
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
def get_svm():
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
    app.run(debug = True, use_reloader = False)