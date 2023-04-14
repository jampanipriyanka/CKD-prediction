import pickle

# import flask
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for

app=Flask(__name__)

import warnings
from warnings import filterwarnings

# import keras
# from xgboost import XGBClassifier
import scipy
# import tensorflow
from sklearn.feature_selection import \
    SelectKBest  # Also known as Information Gain
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

XGBClassifier()



model=pickle.load(open('CKD_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.form.values())
    int_feature=[float(x) for x in request.form.values()]
    print(int_feature)
    int_feature=int_feature[1:]
    Final_features=[np.array(int_feature)]
    predict=model.predict(Final_features)
    print(predict[0])
    return render_template('main.html',prediction_text="{}".format(predict[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)

if __name__== '__main__':
    app.run(debug=True)
    
