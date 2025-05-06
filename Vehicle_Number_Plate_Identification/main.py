import cv2
from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import mysql.connector
from paddleocr import PaddleOCR

import mysql.connector
print("mysql-connector-python installed and working ✅")



class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()  # Initialize speed region
        self.spd = {}  # Dictionary to store speed data
        self.trkd_ids = []  # List for already tracked and speed-estimated IDs
        self.trk_pt = {}  # Dictionary for previous timestamps
        self.trk_pp = {}  # Dictionary for previous positions
        self.logged_ids = set()  # Set to keep track of already logged IDs

        # Initialize the OCR system
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # MySQL database connection
        self.db_connection = self.connect_to_db()

    def connect_to_db(self):
        """Establish connection to MySQL database and create database/table if not exists."""
        try:
            # Connect to MySQL server
            connection = mysql.connector.connect(
                host="", # Replace with your MySQL host (e.g., "localhost")
                user="",  # Replace with your MySQL username
                password=""   # Replace with your MySQL password
            )
            cursor = connection.cursor()

            # Create database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS numberplates_speed")
            print("Database 'numberplates_speed' checked/created.")

            # Connect to the newly created or existing database
            connection.database = "numberplates_speed"

            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS my_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE,
                time TIME,
                track_id INT,
                class_name VARCHAR(255),
                speed FLOAT,
                numberplate TEXT
            )
            """
            cursor.execute(create_table_query)
            print("Table 'my_data' checked/created.")

            return connection
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            raise

    def perform_ocr(self, image_array):
        """Performs OCR on the given image and returns the extracted text."""
        if image_array is None:
            raise ValueError("Image is None")
        if isinstance(image_array, np.ndarray):
            results = self.ocr.ocr(image_array, rec=True)
        else:
            raise TypeError("Input image is not a valid numpy array")

        ocr_text = ' '.join([result[1][0] for result in results[0]] if results[0] else "")
        
        # Print OCR output for debugging
        print("OCR Result: ", ocr_text)  # Check if a number plate is detected
        
        return ocr_text


    def save_to_database(self, date, time, track_id, class_name, speed, numberplate):
        """Save data to the MySQL database."""
        try:
            # Print out the data before saving to check
            print(f"Attempting to save data: {date}, {time}, {track_id}, {class_name}, {speed}, {numberplate}")
            cursor = self.db_connection.cursor()
            query = """
                INSERT INTO my_data (date, time, track_id, class_name, speed, numberplate)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (date, time, track_id, class_name, speed, numberplate))
            self.db_connection.commit()
            print(f"Data saved to database: {date}, {time}, {track_id}, {class_name}, {speed}, {numberplate}")
        except mysql.connector.Error as err:
            print(f"Error saving to database: {err}")
            raise


    def estimate_speed(self, im0):
        """Estimate speed of objects and track them."""
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
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            cropped_image = np.array(im0)[y1:y2, x1:x2]
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

        self.display_output(im0)  # Display output with base class function
        return im0


# Open the video file
cap = cv2.VideoCapture('sample.mp4')

# Define region points for counting
region_points = [(0, 145), (1018, 145)]

# Initialize the object counter
speed_obj = SpeedEstimator(
    region=region_points,
    model="best.pt",  # Replace with your YOLO model file or keep it as it is 
    line_width=2
)

count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Skip odd frames
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Process the frame with the object counter
    result = speed_obj.estimate_speed(frame)

    # Show the frame
    cv2.imshow("RGB", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()