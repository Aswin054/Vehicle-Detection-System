import torch
import cv2
import easyocr
import json
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
from flask import Flask, render_template_string, request, jsonify, send_from_directory

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATH_LP = 'license_plate_detector.pt'
MODEL_PATH_LORRY = 'last.pt'
PLATE_RECORDS_PATH = 'dumpers_db.json'

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VehicleDetector:
    def __init__(self):
        self.lp_model = None
        self.lorry_model = None
        self.reader = None
        self.plate_records = None
        
    def initialize_models(self):
        print("\nInitializing models...")
        
        try:
            self.lp_model = YOLO(MODEL_PATH_LP)
            print(f"✅ License plate model loaded")
        except Exception as e:
            print(f"❌ Error loading license plate model: {str(e)}")
            return False
            
        try:
            self.lorry_model = YOLO(MODEL_PATH_LORRY)
            print(f"✅ Lorry detection model loaded")
        except Exception as e:
            print(f"❌ Error loading lorry model: {str(e)}")
            return False
            
        try:
            self.reader = easyocr.Reader(['en'])
            print("✅ EasyOCR reader initialized")
        except Exception as e:
            print(f"❌ Error initializing EasyOCR: {str(e)}")
            return False
            
        try:
            with open(PLATE_RECORDS_PATH, 'r') as f:
                self.plate_records = json.load(f)
            print(f"✅ Plate records loaded")
        except Exception as e:
            print(f"❌ Error loading plate records: {str(e)}")
            return False
            
        return True

    def detect_vehicles(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        results = {
            'lorries': [],
            'plates': []
        }
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.lorry_model:
            lorry_results = self.lorry_model(frame_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
            for box in lorry_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf.item()
                cls = box.cls.item()
                
                results['lorries'].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Lorry {conf:.2f}", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if self.lp_model:
            lp_results = self.lp_model(frame_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
            for box in lp_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf.item()
                
                plate_text = ""
                cropped_plate = frame[y1:y2, x1:x2]
                if cropped_plate.size > 0 and self.reader:
                    text_results = self.reader.readtext(cropped_plate, detail=0)
                    if text_results:
                        plate_text = ''.join(text_results).replace(' ', '').upper()
                
                status = 'Old' if plate_text and plate_text in self.plate_records.get('license_plates', []) else 'New'
                
                results['plates'].append({
                    'bbox': [x1, y1, x2, y2],
                    'text': plate_text,
                    'status': status,
                    'confidence': conf
                })
                
                color = (0, 0, 255) if status == 'Old' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{plate_text} ({status})" if plate_text else "Plate", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, results

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Detection System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            font-size: 2.2em;
        }
        .upload-section, .results-section {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        .video-wrapper {
            width: 48%;
        }
        video {
            width: 100%;
            border-radius: 4px;
            background-color: #000;
        }
        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .btn:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        .progress-container {
            margin-top: 20px;
        }
        progress {
            width: 100%;
            height: 20px;
        }
        .stats {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .stat-item {
            margin-bottom: 8px;
        }
        .stat-label {
            font-weight: bold;
            display: inline-block;
            width: 180px;
        }
        .detection-results {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 4px;
        }
        .result-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .plate-old {
            color: #e74c3c;
            font-weight: bold;
        }
        .plate-new {
            color: #2ecc71;
            font-weight: bold;
        }
        .status-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Vehicle Detection System</h1>
            <p>Detect lorries and license plates in video files</p>
        </div>
    </header>

    <div class="container">
        <div class="upload-section">
            <h2>Upload Video</h2>
            <input type="file" id="videoInput" accept="video/*">
            <button id="uploadBtn" class="btn">Upload</button>
            <div id="uploadStatus" class="status-message" style="display: none;"></div>
        </div>

        <div class="results-section">
            <h2>Process Video</h2>
            <button id="processBtn" class="btn" disabled>Process Video</button>
            
            <div id="progressContainer" class="progress-container" style="display: none;">
                <progress id="progressBar" value="0" max="100"></progress>
                <div id="progressText">Processing: 0%</div>
            </div>

            <div id="resultsContainer" style="display: none;">
                <div class="video-container">
                    <div class="video-wrapper">
                        <h3>Original Video</h3>
                        <video id="originalVideo" controls></video>
                    </div>
                    <div class="video-wrapper">
                        <h3>Processed Video</h3>
                        <video id="processedVideo" controls></video>
                    </div>
                </div>

                <div class="stats">
                    <h3>Processing Statistics</h3>
                    <div class="stat-item">
                        <span class="stat-label">Total Frames Processed:</span>
                        <span id="totalFrames">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Lorries Detected:</span>
                        <span id="totalLorries">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Plates Detected:</span>
                        <span id="totalPlates">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Unique Plates Found:</span>
                        <span id="uniquePlates">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Processing Time:</span>
                        <span id="processingTime">0</span> seconds
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Processing Speed:</span>
                        <span id="processingFPS">0</span> FPS
                    </div>
                </div>

                <div class="detection-results">
                    <h3>Detection Results</h3>
                    <div id="resultsList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFilename = '';
        
        document.getElementById('uploadBtn').addEventListener('click', async () => {
            const fileInput = document.getElementById('videoInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const uploadStatus = document.getElementById('uploadStatus');
            
            if (!fileInput.files || fileInput.files.length === 0) {
                showStatus('Please select a video file first', 'error');
                return;
            }
            
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            
            uploadBtn.disabled = true;
            uploadStatus.textContent = 'Uploading video...';
            uploadStatus.style.display = 'block';
            uploadStatus.className = 'status-message';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentFilename = data.filename;
                    document.getElementById('processBtn').disabled = false;
                    showStatus(data.message, 'success');
                    
                    // Show original video
                    const originalVideo = document.getElementById('originalVideo');
                    originalVideo.src = `/uploads/${currentFilename}`;
                } else {
                    showStatus(data.error, 'error');
                }
            } catch (error) {
                showStatus('Error uploading file: ' + error.message, 'error');
            } finally {
                uploadBtn.disabled = false;
            }
        });
        
        document.getElementById('processBtn').addEventListener('click', async () => {
            const processBtn = document.getElementById('processBtn');
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            
            processBtn.disabled = true;
            progressContainer.style.display = 'block';
            progressBar.value = 0;
            progressText.textContent = 'Processing: 0%';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        filename: currentFilename,
                        output_video: 'processed_' + currentFilename
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Simulate progress updates (in a real app, you'd use WebSockets or polling)
                    simulateProgress(progressBar, progressText, () => {
                        showResults(data);
                    });
                } else {
                    showStatus(data.error, 'error');
                    progressContainer.style.display = 'none';
                    processBtn.disabled = false;
                }
            } catch (error) {
                showStatus('Error processing video: ' + error.message, 'error');
                progressContainer.style.display = 'none';
                processBtn.disabled = false;
            }
        });
        
        function simulateProgress(progressBar, progressText, callback) {
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    setTimeout(callback, 500);
                }
                progressBar.value = progress;
                progressText.textContent = `Processing: ${Math.round(progress)}%`;
            }, 300);
        }
        
        function showResults(data) {
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'block';
            
            // Show processed video
            const processedVideo = document.getElementById('processedVideo');
            processedVideo.src = `/output/${data.output_video}`;
            
            // Update stats
            document.getElementById('totalFrames').textContent = data.stats.total_frames;
            document.getElementById('totalLorries').textContent = data.stats.total_lorries;
            document.getElementById('totalPlates').textContent = data.stats.total_plates;
            document.getElementById('uniquePlates').textContent = data.stats.unique_plates;
            document.getElementById('processingTime').textContent = data.stats.processing_time;
            document.getElementById('processingFPS').textContent = data.stats.processing_fps;
            
            // Load detailed results
            fetch(`/output/${data.results_file}`)
                .then(response => response.json())
                .then(results => {
                    const resultsList = document.getElementById('resultsList');
                    resultsList.innerHTML = '';
                    
                    // Show sample of results (first 20 frames with detections)
                    const sampleResults = results.filter(r => 
                        r.results.lorries.length > 0 || r.results.plates.length > 0
                    ).slice(0, 20);
                    
                    sampleResults.forEach(frame => {
                        const frameDiv = document.createElement('div');
                        frameDiv.className = 'result-item';
                        
                        let content = `<strong>Frame ${frame.frame}</strong> (${frame.timestamp.toFixed(2)}s): `;
                        
                        if (frame.results.lorries.length > 0) {
                            content += `Lorries: ${frame.results.lorries.length} `;
                        }
                        
                        if (frame.results.plates.length > 0) {
                            content += `Plates: `;
                            frame.results.plates.forEach(plate => {
                                const statusClass = plate.status === 'Old' ? 'plate-old' : 'plate-new';
                                content += `<span class="${statusClass}">${plate.text || 'Unknown'} (${plate.status})</span>, `;
                            });
                        }
                        
                        if (frame.results.lorries.length === 0 && frame.results.plates.length === 0) {
                            content += 'No detections';
                        }
                        
                        frameDiv.innerHTML = content.replace(/, $/, '');
                        resultsList.appendChild(frameDiv);
                    });
                });
            
            showStatus('Video processing completed successfully!', 'success');
            document.getElementById('processBtn').disabled = false;
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${type}`;
            statusDiv.style.display = 'block';
        }
    </script>
</body>
</html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(UPLOAD_FOLDER, f"upload_{timestamp}.mp4")
        file.save(video_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': f"upload_{timestamp}.mp4"
        }), 200

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    filename = data.get('filename')
    output_video = data.get('output_video', 'processed.mp4')
    
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, output_video)
    output_json = os.path.join(OUTPUT_FOLDER, 'results.json')

    detector = VehicleDetector()
    if not detector.initialize_models():
        return jsonify({'error': 'Failed to initialize models'}), 500

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video'}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_results = []
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if frame_count % FRAME_SKIP != 0:
            continue

        processed_frame, results = detector.detect_vehicles(frame)
        all_results.append({
            'frame': frame_count,
            'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
            'results': results
        })
        out.write(processed_frame)

    cap.release()
    out.release()

    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=4)

    processing_time = time.time() - start_time
    total_lorries = sum(len(frame['results']['lorries']) for frame in all_results)
    total_plates = sum(len(frame['results']['plates']) for frame in all_results)
    unique_plates = len(set(p['text'] for frame in all_results for p in frame['results']['plates'] if p['text']))

    return jsonify({
        'message': 'Processing complete',
        'output_video': output_video,
        'results_file': 'results.json',
        'stats': {
            'total_frames': frame_count,
            'total_lorries': total_lorries,
            'total_plates': total_plates,
            'unique_plates': unique_plates,
            'processing_time': round(processing_time, 2),
            'processing_fps': round(frame_count/max(processing_time, 0.1), 1)
        }
    })

@app.route('/output/<filename>')
def get_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)