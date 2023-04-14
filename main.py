import numpy as np
import pandas as pd
from flask import Flask,request ,jsonify, render_template,url_for
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import warnings
from warnings import filterwarnings
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
# import tensorflow
# from xgboost import XGBClassifier
import scipy
from sklearn.feature_selection import SelectKBest#Also known as Information Gain
from sklearn.feature_selection import chi2
# import xgboost
from xgboost import XGBClassifier


app=Flask(__name__)


model=pickle.load(open('CKD_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feature=[float(x) for x in request.form.values()]
    Final_features=[np.array(int_feature)]
    predict=model.predict(Final_features)
    print(predict[0])
    return render_template('home.html',prediction_text="{}".format(predict[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)

if __name__== '__main__':
    app.run(debug=True)
    
