from flask import Flask, jsonify, request
#from sklearn.externals import joblib
import joblib
from flask import Flask, render_template, flash, request
app = Flask(__name__,template_folder='template')
@app.route('/')

def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])

def predict ():
    model_load = joblib.load('wolken.pkl')
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        prediction = model_load.predict(data)
    return jsonify({'prediction' : prediction[0]})



if __name__ == '__main__':
	app.run(debug=True)