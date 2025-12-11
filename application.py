import pickle
from flask import Flask, request, render_template, jsonify

application = Flask(__name__)
app = application

# Correct path
carmodel = pickle.load(open(r'C:\Users\gupje\Desktop\ml\ml project_car\modal\regression_car_price_prediction.pkl', 'rb'))
standard_scaler = pickle.load(open(r'C:\Users\gupje\Desktop\ml\ml project_car\modal\scaler_car_price_prediction.pkl', 'rb'))

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
