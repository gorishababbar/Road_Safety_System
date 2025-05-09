import cv2
from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import sqlite3
import os
from paddleocr import PaddleOCR
import tempfile

# Disable excessive logging from PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()  # Initialize speed region
        self.spd = {}  # Dictionary to store speed data
        self.trkd_ids = []  # List for already tracked and speed-estimated IDs
        self.trk_pt = {}  # Dictionary for previous timestamps
        self.trk_pp = {}  # Dictionary for previous positions
        self.logged_ids = set()  # Set to keep track of already logged IDs
        self.detected_plates = []  # Store detected plates for return

        # Initialize the OCR system
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # SQLite database connection
        self.db_connection = self.connect_to_db()

    def connect_to_db(self):
        """Establish connection to SQLite database and create table if not exists."""
        try:
            # Get database path in the same directory as the script
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numberplates_speed.db')
            
            # Connect to SQLite database (creates it if it doesn't exist)
            connection = sqlite3.connect(db_path)
            cursor = connection.cursor()

            # Create table if it doesn't exist - SQLite syntax
            create_table_query = """
            CREATE TABLE IF NOT EXISTS my_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                track_id INTEGER,
                class_name TEXT,
                speed REAL,
                numberplate TEXT
            )
            """
            cursor.execute(create_table_query)
            connection.commit()
            return connection
        except sqlite3.Error as err:
            print(f"Error connecting to database: {err}")
            raise

    def perform_ocr(self, image_array):
        """Performs OCR on the given image and returns the extracted text."""
        try:
            if image_array is None:
                return ""
            if isinstance(image_array, np.ndarray) and image_array.size > 0:
                results = self.ocr.ocr(image_array, rec=True)
                if results and results[0]:
                    ocr_text = ' '.join([result[1][0] for result in results[0]])
                    return ocr_text
            return ""
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def save_to_database(self, date, time, track_id, class_name, speed, numberplate):
        """Save data to the SQLite database."""
        try:
            cursor = self.db_connection.cursor()
            query = """
                INSERT INTO my_data (date, time, track_id, class_name, speed, numberplate)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (date, time, track_id, class_name, speed, numberplate))
            self.db_connection.commit()
            
            # Also store in our results list
            self.detected_plates.append({
                'date': date,
                'time': time,
                'track_id': track_id,
                'class_name': class_name,
                'speed': speed,
                'numberplate': numberplate
            })
            
        except sqlite3.Error as err:
            print(f"Error saving to database: {err}")
            raise

    def estimate_speed(self, im0):
        """Estimate speed of objects and track them."""
        try:
            self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
            self.extract_tracks(im0)  # Extract tracks

            # Get current date and time
            current_time = datetime.now()

            for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
                self.store_tracking_history(track_id, box)  # Store track history

                if track_id not in self.trk_pt:
                    self.trk_pt[track_id] = 0
                if track_id not in self.trk_pp:
                    self.trk_pp[track_id] = self.track_line[-1]

                speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]

                # Draw the bounding box and track ID on it
                label = f"ID: {track_id} {speed_label}"  # Show track ID along with speed
                self.annotator.box_label(box, label=label, color=colors(track_id, True))  # Draw bounding box

                # Speed and direction calculation
                if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                    direction = "known"
                else:
                    direction = "unknown"

                # Calculate speed if the direction is known and the object is new
                if direction == "known" and track_id not in self.trkd_ids:
                    self.trkd_ids.append(track_id)
                    time_difference = time() - self.trk_pt[track_id]
                    if time_difference > 0:
                        speed = np.abs(self.track_line[-1][1].item() - self.trk_pp[track_id][1].item()) / time_difference
                        self.spd[track_id] = round(speed)

                # Update the previous tracking time and position
                self.trk_pt[track_id] = time()
                self.trk_pp[track_id] = self.track_line[-1]
                
                try:
                    x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 >= im0.shape[1]: x2 = im0.shape[1] - 1
                    if y2 >= im0.shape[0]: y2 = im0.shape[0] - 1
                    
                    if x2 > x1 and y2 > y1:
                        cropped_image = np.array(im0)[y1:y2, x1:x2].copy()
                        ocr_text = self.perform_ocr(cropped_image)

                        # Get the class name and speed
                        class_name = self.names[int(cls)]
                        speed = self.spd.get(track_id)

                        # Ensure OCR text is not empty and save OCR text with the relevant details if not already logged
                        if track_id not in self.logged_ids and ocr_text.strip() and speed is not None:
                            self.save_to_database(
                                current_time.strftime("%Y-%m-%d"),
                                current_time.strftime("%H:%M:%S"),
                                track_id,
                                class_name,
                                speed if speed is not None else 0.0,
                                ocr_text
                            )
                            self.logged_ids.add(track_id)
                except Exception as e:
                    print(f"Error processing vehicle crop: {e}")

            self.display_output(im0)  # Display output with base class function
            return im0
        except Exception as e:
            print(f"Error in estimate_speed: {e}")
            return im0

def process_video_for_plates(video_path, output_folder):
    """Process a video file to detect license plates."""
    results = {
        'plates': [],
        'processed_video': None,
        'error': None
    }
    
    try:
        # Define region points for counting
        region_points = [(0, 145), (1018, 145)]

        # Initialize the speed estimator
        speed_obj = SpeedEstimator(
            region=region_points,
            model="yolov8n.pt",  # Using standard YOLOv8 model
            line_width=2
        )
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            results['error'] = "Could not open video file"
            return results
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_filename = f"processed_{os.path.basename(video_path)}"
        output_path = os.path.join(output_folder, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))
        
        count = 0
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            count += 1
            if count % 10 != 0:  # Process every 10th frame
                continue
                
            frame = cv2.resize(frame, (640, 360))
            
            # Process the frame
            result = speed_obj.estimate_speed(frame)
            
            # Write to output video
            out.write(result)
        
        # Release resources
        cap.release()
        out.release()
        
        # Get detected plates from the speed estimator
        results['plates'] = speed_obj.detected_plates
        results['processed_video'] = output_filename
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def get_detected_plates():
    """Get all detected license plates from the database."""
    plates = []
    try:
        # Get database path
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numberplates_speed.db')
        
        # Connect to SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Query all data
        cursor.execute("SELECT * FROM my_data")
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        for row in rows:
            plates.append({
                'id': row[0],
                'date': row[1],
                'time': row[2],
                'track_id': row[3],
                'class_name': row[4],
                'speed': row[5],
                'numberplate': row[6]
            })
        
        connection.close()
    except Exception as e:
        print(f"Error retrieving plates: {e}")
    
    return plates