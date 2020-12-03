from pred_model import make_prediction
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tick = [request.form['tick']]
    prediction = make_prediction(tick)

    return render_template('index.html', prediction_text='Predicted price after 30 days: ₹{}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
