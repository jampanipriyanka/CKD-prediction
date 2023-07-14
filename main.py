import pickle

# import flask
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for

app=Flask(__name__)


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier



model=pickle.load(open('CKD_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    items=request.form.items()
    print(items)
    x_train_head=['age', 'Gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco','active']
    int_feature={key:float(value) for key,value in request.form.items() }
    print(int_feature) 
    trainvalues={x:int_feature[x] for x in x_train_head}
    print(trainvalues)
    df=pd.DataFrame(trainvalues,x_train_head)
    predict=model.predict(df)
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
    
