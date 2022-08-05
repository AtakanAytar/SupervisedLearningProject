
from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
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
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)