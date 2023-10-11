import joblib
from flask import Flask,request,render_template
import numpy as np

app = Flask(__name__)
model = joblib.load(open('wine_quality_lgbm_model.pkl','rb'))


@app.route('/')

def home():
  result = ''
  return render_template('index.html',**locals)

@app.route('/predict',methods=['POST','GET'])
def predict():
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        feature_vector = np.array([[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]])
        prediction = model.predict(feature_vector)
        return render_template('index.html', prediction = prediction)
if __name__ == '__main__':
    app.run(debug=True)