import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import sys
from werkzeug.utils import secure_filename

# Add the Video_Summarization directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Video_Summarization'))

# Add the Video_Authenticity_Detection directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Video_Authenticity_Detection'))
from main_refactor import detect_video_forgery

# Import the video summarization function
from video_summarization_refactor import summarize_video

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Vehicle_Number_Plate_Identification'))
from numberPlateRefactor import process_video_for_plates, get_detected_plates

app = Flask(__name__)
app.secret_key = 'road_safety_system_2025'  # required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Homepage with buttons for all modules"""
    return render_template('index.html')

@app.route('/video-authentication', methods=['GET', 'POST'])
def video_authentication():
    """Video Authenticity Detection module"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['video']
        
        # If user did not select a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Get threshold value from form
        threshold = request.form.get('threshold', 3, type=int)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the video
            try:
                results = detect_video_forgery(file_path, threshold, app.config['UPLOAD_FOLDER'])
                
                # Check for errors
                if "error" in results:
                    flash(f"Error: {results['error']}")
                    return redirect(request.url)
                
                # Return results to the template
                return render_template('video_auth.html', 
                                       results=results, 
                                       has_results=True, 
                                       video_filename=filename)
                
            except Exception as e:
                flash(f'Error analyzing video: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload MP4, AVI, MOV, or MKV.')
            return redirect(request.url)
            
    return render_template('video_auth.html', has_results=False)

@app.route('/video-summarization', methods=['GET', 'POST'])
def video_summarization():
    """Video Summarization module"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['video']
        
        # If user did not select a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Generate an output path for the summary
            summary_filename = f"summary_{filename.rsplit('.', 1)[0]}.txt"
            summary_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_filename)
            
            # Process the video
            try:
                summary = summarize_video(file_path, summary_path)
                return render_template('video_sum.html', summary=summary, 
                                      video_filename=filename, has_results=True)
            except Exception as e:
                flash(f'Error processing video: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload MP4, AVI, MOV, or MKV.')
            return redirect(request.url)
            
    return render_template('video_sum.html', has_results=False)

@app.route('/criminal-activity')
def criminal_activity():
    """Criminal Activity Detection module"""
    return render_template('criminal.html')

@app.route('/traffic-anomaly')
def traffic_anomaly():
    """Traffic Anomaly Detection module"""
    return render_template('traffic.html')


@app.route('/vehicle-number-plate', methods=['GET', 'POST'])
def vehicle_number_plate():
    """Vehicle Number Plate Identification module"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['video']
        
        # If user did not select a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Get speed limit from form
        try:
            speed_limit = int(request.form.get('speed_limit', 60))  # Default to 60 km/h
        except ValueError:
            speed_limit = 60  # Default if invalid input
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the video for license plates
            try:
                results = process_video_for_plates(file_path, app.config['UPLOAD_FOLDER'])
                
                # Check for errors
                if results['error']:
                    flash(f"Error: {results['error']}")
                    return redirect(request.url)
                
                # Get all historical plates from database
                all_plates = get_detected_plates()
                
                # Return results to the template
                return render_template('vehicle.html', 
                                      results=results,
                                      all_plates=all_plates,
                                      has_results=True, 
                                      video_filename=filename,
                                      processed_video=results['processed_video'],
                                      speed_limit=speed_limit)  # Pass speed limit to template
                
            except Exception as e:
                flash(f'Error processing video: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload MP4, AVI, MOV, or MKV.')
            return redirect(request.url)
            
    return render_template('vehicle.html', has_results=False, speed_limit=60)  # Default speed limit

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from the upload folder"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    # Disable auto-reloading to prevent TensorFlow-related restart issues
    app.run(debug=False)