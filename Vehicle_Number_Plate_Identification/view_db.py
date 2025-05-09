# Save as view_database.py in the same folder as your main.py
import sqlite3
import os

def view_database():
    # Connect to the database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numberplates_speed.db')
    
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        return
        
    print(f"Opening database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get number of records
    cursor.execute("SELECT COUNT(*) FROM my_data")
    count = cursor.fetchone()[0]
    print(f"Total records: {count}")
    
    # Query all data from the table
    cursor.execute("SELECT * FROM my_data")
    rows = cursor.fetchall()
    
    # Print column headers
    print("\n--- License Plate Detections ---")
    print("ID | Date | Time | Track ID | Vehicle Type | Speed | License Plate")
    print("-" * 80)
    
    # Print each row
    for row in rows:
        print(" | ".join(str(item) for item in row))
    
    conn.close()

if __name__ == "__main__":
    view_database()