from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    """Homepage with buttons for all modules"""
    return render_template('index.html')

@app.route('/video-authentication')
def video_authentication():
    """Video Authenticity Detection module"""
    return render_template('video_auth.html')

@app.route('/video-summarization')
def video_summarization():
    """Video Summarization module"""
    return render_template('video_sum.html')

@app.route('/criminal-activity')
def criminal_activity():
    """Criminal Activity Detection module"""
    return render_template('criminal.html')

@app.route('/traffic-anomaly')
def traffic_anomaly():
    """Traffic Anomaly Detection module"""
    return render_template('traffic.html')

@app.route('/vehicle-number-plate')
def vehicle_number_plate():
    """Vehicle Number Plate Identification module"""
    return render_template('vehicle.html')

if __name__ == '__main__':
    app.run(debug=True)