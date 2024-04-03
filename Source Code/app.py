import flask
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import pickle
from xgboost import XGBClassifier

app = Flask(__name__, template_folder="templates")
model = pickle.load(open("model_rf.pkl", "rb"))
print("Model Loaded")

def validate_input(date, minTemp, maxTemp, rainfall, evaporation, sunshine,
                  windGustSpeed, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm,
                  pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm,
                  location, windDir9am, windDir3pm, windGustDir, rainToday):
    errors = []

    # Validate date
    try:
        # Convert date to datetime object
        pd.to_datetime(date)
    except ValueError:
        # Try to convert date to datetime object with "DD-MM-YYYY" format
        try:
            pd.to_datetime(date, format="%d-%m-%Y")
        except ValueError:
            errors.append("Invalid date format. Please use DD-MM-YYYY or YYYY-MM-DD.")

    # Validate minTemp
    if not -50 <= minTemp <= 50:
        errors.append("MinTemp must be between -50 and 50.")

    # Validate maxTemp
    if not -50 <= maxTemp <= 50:
        errors.append("MaxTemp must be between -50 and 50.")

    # Validate rainfall
    if not 0 <= rainfall <= 50:
        errors.append("Rainfall must be between 0 and 50.")# Measured in millimeters

    # Validate evaporation
    if not 0 <= evaporation <= 15:
        errors.append("Evaporation must be between 0 and 15.")# Measured in millimeters per day

    # Validate sunshine
    if not 0 <= sunshine <= 24:
        errors.append("Sunshine must be between 0 and 24.")# Hours per day

    # Validate windGustSpeed
    if not 0 <= windGustSpeed <= 100:
        errors.append("Wind Gust Speed must be between 0 and 100.")# km/h

    # Validate windSpeed9am
    if not 0 <= windSpeed9am <= 100:
        errors.append("Wind Speed 9am must be between 0 and 100.")# km/h

    # Validate windSpeed3pm
    if not 0 <= windSpeed3pm <= 100:
        errors.append("Wind Speed 3pm must be between 0 and 100.")# km/h

    # Validate humidity9am
    if not 0 <= humidity9am <= 100:
        errors.append("Humidity 9am must be between 0 and 100.")# Percentage

    # Validate humidity3pm
    if not 0 <= humidity3pm <= 100:
        errors.append("Humidity 3pm must be between 0 and 100.")# Percentage

    # Validate pressure9am
    if not 900 <= pressure9am <= 1100:
        errors.append("Pressure 9am must be between 900 and 1100.")# hPa

    # Validate pressure3pm
    if not 900 <= pressure3pm <= 1100:
        errors.append("Pressure 3pm must be between 900 and 1100.")# hPa

    # Validate temp9am
    if not -50 <= temp9am <= 50:
        errors.append("Temperature 9am must be between -50 and 50.")

    # Validate temp3pm
    if not -50 <= temp3pm <= 50:
        errors.append("Temperature 3pm must be between -50 and 50.")

    # Validate cloud9am
    if not 0 <= cloud9am <= 1:
        errors.append("Cloud 9am must be between 0 and 1.")

    # Validate cloud3pm
    if not 0 <= cloud3pm <= 1:
        errors.append("Cloud 3pm must be between 0 and 1.")

    # Remove non-ASCII characters
    for s in [location, windDir9am, windDir3pm, windGustDir, rainToday]:
        s.encode('ascii', 'ignore')
    return errors

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            print("Form data received:")
            for key, value in request.form.items():
                print(f"{key}: {value}")
            date = request.form['date']
            minTemp = float(request.form['mintemp'])
            maxTemp = float(request.form['maxtemp'])
            rainfall = float(request.form['rainfall'])
            evaporation = float(request.form['evaporation'])
            sunshine = float(request.form['sunshine'])
            windGustSpeed = float(request.form['windgustspeed'])
            windSpeed9am = float(request.form['windspeed9am'])
            windSpeed3pm = float(request.form['windspeed3pm'])
            humidity9am = float(request.form['humidity9am'])
            humidity3pm = float(request.form['humidity3pm'])
            pressure9am = float(request.form['pressure9am'])
            pressure3pm = float(request.form['pressure3pm'])
            temp9am = float(request.form['temp9am'])
            temp3pm = float(request.form['temp3pm'])
            cloud9am = int(request.form['cloud9am'])
            cloud3pm = int(request.form['cloud3pm'])
            location = request.form['location']
            windDir9am = request.form['winddir9am']
            windDir3pm = request.form['winddir3pm']
            windGustDir = request.form['windgustdir']
            rainToday = request.form['raintoday']
            

            # Validate input
            errors = validate_input(
                date, minTemp, maxTemp, rainfall, evaporation, sunshine,
                windGustSpeed, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm,
                pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm,
                location, windDir9am, windDir3pm, windGustDir, rainToday
            )
            
            if errors:
                error_message = "Validation errors: " + ", ".join(errors)
                print(error_message)
                return render_template("error.html", error_message=error_message)

            inplst = [
                    location if var_name == 'location' else var_val
                    for var_name, var_val in zip(
                        ['location', 'minTemp', 'maxTemp', 'rainfall', 'evaporation', 'sunshine',
                         'windGustDir', 'windGustSpeed', 'windDir9am', 'windDir3pm',
                         'windSpeed9am', 'windSpeed3pm',
                         'humidity9am', 'humidity3pm', 'pressure9am', 'pressure3pm',
                         'cloud9am', 'cloud3pm', 'temp9am', 'temp3pm',
                         'rainToday'],
                        [location, minTemp, maxTemp, rainfall, evaporation, sunshine,
                         windGustDir, windGustSpeed, windDir9am, windDir3pm,
                         windSpeed9am, windSpeed3pm,
                         humidity9am, humidity3pm, pressure9am, pressure3pm,
                         cloud9am, cloud3pm, temp9am, temp3pm,
                         rainToday]
                    )
                ]
            pred = model.predict([inplst])
            output = pred[0]

            if output == 0:
                return render_template("rainy.html", prediction="It's Sunny!")
            else:
                return render_template("sunny.html", prediction="It's Rainy!")

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run()
