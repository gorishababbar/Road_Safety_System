import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import time
from collections import Counter
import matplotlib.pyplot as plt
import argparse

# Constants
RESOLUTION = 224
CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal"]

# Model architecture
class CrimeModelCNN(nn.Module):
    def __init__(self):
        super(CrimeModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding="same")
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.max_pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 256, 256)
        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.max_pool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout4(x)
        return x

class CrimeModelLSTM(nn.Module):
    def __init__(self):
        super(CrimeModelLSTM, self).__init__()
        self.lstm1 = nn.LSTM(1, 8, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(8, 8, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(8, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.dropout(x)
        return x

class CrimeModel(nn.Module):
    def __init__(self):
        super(CrimeModel, self).__init__()
        self.cnn = CrimeModelCNN()
        self.lstm = CrimeModelLSTM()
        self.fc = nn.Linear(260, 8)

    def forward(self, x):
        x_cnn = x
        x_lstm = torch.reshape(x, (x.shape[0], RESOLUTION * RESOLUTION, 1))
        x_cnn = self.cnn(x_cnn)
        x_lstm = self.lstm(x_lstm)
        x_combined = torch.cat((x_cnn, x_lstm), dim=1)
        x = self.fc(x_combined)
        return x

def preprocess_image(image_path):
    """Preprocess an image for the crime detection model"""
    try:
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to YUV color space and extract Y channel
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel, _, _ = cv2.split(img_yuv)
        
        # Create transformer
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize((RESOLUTION, RESOLUTION))
        ])
        
        # Transform the image
        tensor = transformer(y_channel)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def detect_criminal_activity(frames_dir, model_path):
    """
    Detect criminal activity from extracted video frames
    
    Parameters:
    -----------
    frames_dir : str
        Directory containing extracted frames
    model_path : str
        Path to the trained model file
        
    Returns:
    --------
    dict
        Results including detected activity and confidence
    """
    results = {
        'success': False,
        'predicted_class': None,
        'confidence': 0,
        'class_distribution': {},
        'error': None
    }
    
    try:
        # Check frames directory
        frames_dir = Path(frames_dir)
        if not frames_dir.exists() or not frames_dir.is_dir():
            results['error'] = f"Frames directory not found: {frames_dir}"
            return results
        
        # Check model file
        model_path = Path(model_path)
        if not model_path.exists():
            results['error'] = f"Model file not found: {model_path}"
            return results
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model
        print("Loading model...")
        model = CrimeModel()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
        
        # Get all image files
        image_files = sorted(list(frames_dir.glob("*.jpg")))
        if not image_files:
            results['error'] = f"No image files found in {frames_dir}"
            return results
        
        print(f"Found {len(image_files)} frames to process")
        
        # Process frames
        predictions = []
        batch_size = 16
        
        start_time = time.time()
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_tensors = []
            
            # Preprocess each image
            for img_path in batch_files:
                tensor = preprocess_image(img_path)
                if tensor is not None:
                    batch_tensors.append(tensor)
            
            if not batch_tensors:
                continue
            
            # Concatenate and process batch
            batch = torch.cat(batch_tensors, dim=0).to(device)
            
            with torch.no_grad():
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                
                # Get predictions
                for j in range(len(batch_tensors)):
                    pred_class = probs[j].argmax().item()
                    confidence = probs[j][pred_class].item()
                    predictions.append((CLASSES[pred_class], confidence))
            
            # Print progress
            print(f"Processed {min(i+batch_size, len(image_files))}/{len(image_files)} frames", end="\r")
        
        processing_time = time.time() - start_time
        
        # Analyze predictions
        if predictions:
            # Count class occurrences
            class_counts = Counter([p[0] for p in predictions])
            total_frames = len(predictions)
            
            # Get average confidence per class
            class_confidences = {}
            for class_name in class_counts:
                confidences = [p[1] for p in predictions if p[0] == class_name]
                class_confidences[class_name] = sum(confidences) / len(confidences)
            
            # Determine the most common class
            most_common_class, count = class_counts.most_common(1)[0]
            confidence = class_confidences[most_common_class]
            
            # Create class distribution dictionary
            class_distribution = {}
            for class_name, count in class_counts.items():
                class_distribution[class_name] = {
                    'count': count,
                    'percentage': count / total_frames * 100,
                    'avg_confidence': class_confidences.get(class_name, 0)
                }
            
            # Create visualization
            create_visualization(frames_dir, predictions, class_distribution)
            
            # Update results
            results['success'] = True
            results['predicted_class'] = most_common_class
            results['confidence'] = confidence
            results['class_distribution'] = class_distribution
            results['processing_time'] = processing_time
            results['frame_count'] = len(image_files)
        else:
            results['error'] = "No predictions made"
        
    except Exception as e:
        import traceback
        results['error'] = str(e)
        print(f"Error detecting criminal activity: {e}")
        print(traceback.format_exc())
    
    return results

def create_visualization(frames_dir, predictions, class_distribution):
    """Create visualization of the results"""
    try:
        # Create output directory
        output_dir = frames_dir / "results"
        output_dir.mkdir(exist_ok=True)
        
        # Create pie chart of class distribution
        classes = list(class_distribution.keys())
        counts = [class_distribution[c]['count'] for c in classes]
        
        plt.figure(figsize=(10, 6))
        plt.pie(
            counts,
            labels=[f"{c} ({class_distribution[c]['count']})" for c in classes],
            autopct='%1.1f%%',
            startangle=90
        )
        plt.axis('equal')
        plt.title('Criminal Activity Detection Results')
        
        # Save chart
        plt.savefig(str(output_dir / "classification_results.png"))
        plt.close()
        
        print(f"Visualization saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect criminal activity from video frames")
    parser.add_argument("frames_dir", help="Directory containing extracted frames")
    parser.add_argument("--model", default="crime_model_epoch_5.pth", 
                        help="Path to trained model (default: crime_model_epoch_5.pth)")
    
    args = parser.parse_args()
    
    print(f"Analyzing frames in {args.frames_dir} using model {args.model}")
    results = detect_criminal_activity(args.frames_dir, args.model)
    
    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print("\n----- Criminal Activity Detection Results -----")
        print(f"Predicted activity: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2f}")
        print("\nClass distribution:")
        
        for class_name, stats in results['class_distribution'].items():
            print(f"  {class_name}: {stats['count']} frames ({stats['percentage']:.1f}%), "
                  f"avg confidence: {stats['avg_confidence']:.2f}")
        
        print(f"\nProcessed {results['frame_count']} frames in {results['processing_time']:.2f} seconds")