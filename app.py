# Importing essential libraries
from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle

# Load the SVM Classifier model
filename = 'LR.pkl'
classifier = pickle.load(open(filename, 'rb'))

app=Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age=int(request.form['age'])
        gender=int(request.form['gender'])
        height=float(request.form['height'])
        weight=float(request.form['weight'])
        ap_hi=int(request.form['ap_hi'])
        ap_lo=int(request.form['ap_lo'])
        cholesterol=int(request.form['cholesterol'])
        gluc=int(request.form['gluc'])
        smoke=int(request.form['smoke'])
        alco=int(request.form['alco'])
        active=int(request.form['active'])

        data = np.array([[age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active]])
        my_prediction=classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)




if __name__=='__main__':
    app.run(debug=True)
