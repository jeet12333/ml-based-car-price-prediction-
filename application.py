import os
import pickle
from flask import Flask, request, render_template, jsonify

application = Flask(__name__)
app = application

# Load model and scaler using relative paths so deployment can find them
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modal', 'regression_car_price_prediction.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'modal', 'scaler_car_price_prediction.pkl')

# fallback: if files are missing, raise a clear error during startup
try:
    carmodel = pickle.load(open(MODEL_PATH, 'rb'))
    standard_scaler = pickle.load(open(SCALER_PATH, 'rb'))
except Exception as e:
    raise RuntimeError(f"Model files not found or failed to load: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":

        doornumber = float(request.form.get('doornumber'))
        drivewheel = float(request.form.get('drivewheel'))
        enginelocation = float(request.form.get('enginelocation'))
        wheelbase = float(request.form.get('wheelbase'))
        carlength = float(request.form.get('carlength'))
        carwidth = float(request.form.get('carwidth'))
        carheight = float(request.form.get('carheight'))
        curbweight = float(request.form.get('curbweight'))
        cylindernumber = float(request.form.get('cylindernumber'))
        enginesize = float(request.form.get('enginesize'))
        boreratio = float(request.form.get('boreratio'))
        stroke = float(request.form.get('stroke'))
        compressionratio = float(request.form.get('compressionratio'))
        horsepower = float(request.form.get('horsepower'))
        peakrpm = float(request.form.get('peakrpm'))
        citympg = float(request.form.get('citympg'))
        highwaympg = float(request.form.get('highwaympg'))

        new_data_scaled = standard_scaler.transform([[
            doornumber, drivewheel, enginelocation, wheelbase, carlength,
            carwidth, carheight, curbweight, cylindernumber, enginesize,
            boreratio, stroke, compressionratio, horsepower, peakrpm,
            citympg, highwaympg
        ]])

        result = carmodel.predict(new_data_scaled)
        return render_template('home.html', results=result[0])

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
