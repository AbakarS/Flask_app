from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


app = Flask(__name__)
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler_customer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    print("intial values -->", int_features)
    pre_final_features = [np.array(int_features)]
    final_features = scaler.transform(pre_final_features)
    print("scaled values -->", final_features)
    prediction = model.predict(final_features)   
    print('predictio value is ', prediction[0])
    if (prediction[0] == 1):
        output = "Cet utilisateur achètera le produit après l'avoir consulté sur les réseaux sociaux"
    elif(prediction[0] == 0):
        output = "Cet utilisateur n'achètera pas le produit après l'avoir consulté sur les réseaux sociaux "
    else:
        output = "Not sure"
        

    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")