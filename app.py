from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import os
import base64

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load model and initialize variables
model = tf.keras.models.load_model("model_new_final.h5")
mp_holistic = mp.solutions.holistic

# Define actions
actions = np.array(["Book","Do","Eat","Go","Good","Hello","Home","Hungry","I","Morning",
                   "No","Not","Pizza","Place","Read","School","Student","Teacher",
                   "Thank You","This","Tomorrow","Want","What","Yes","Yesterday","You"])

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sign Language Detection</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                min-height: 100vh;
                background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: rgba(255, 255, 255, 0.95);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                max-width: 900px;
                width: 100%;
                backdrop-filter: blur(10px);
            }
            
            h1 {
                color: #1a2a6c;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .tabs {
                display: flex;
                justify-content: center;
                margin-bottom: 30px;
                gap: 20px;
            }
            
            .tab {
                padding: 12px 30px;
                background: #1a2a6c;
                color: white;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            
            .tab.active {
                background: #b21f1f;
                transform: scale(1.05);
            }
            
            .tab-content {
                display: none;
                animation: fadeIn 0.5s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .tab-content.active {
                display: block;
            }
            
            .camera-container {
                position: relative;
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            }
            
            #camera-feed {
                width: 100%;
                height: auto;
                display: block;
            }
            
            .camera-controls {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 30px 0;
                flex-wrap: wrap;
            }
            
            .control-btn {
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .control-btn i {
                font-size: 1.2em;
            }
            
            #start-camera {
                background: #1a2a6c;
                color: white;
            }
            
            #record {
                background: #b21f1f;
                color: white;
            }
            
            #stop-record {
                background: #fdbb2d;
                color: #1a2a6c;
                display: none;
            }
            
            #process-recording {
                background: #4CAF50;
                color: white;
            }
            
            .control-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }
            
            .control-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .recording-indicator {
                position: absolute;
                top: 20px;
                right: 20px;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #b21f1f;
                display: none;
            }
            
            .recording-indicator.active {
                display: block;
                animation: pulse 1s infinite;
            }
            
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.2); opacity: 0.5; }
                100% { transform: scale(1); opacity: 1; }
            }
            
            .timer {
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.9em;
                display: none;
            }
            
            .timer.active {
                display: block;
            }
            
            .result {
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-top: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .prediction {
                font-size: 2.5em;
                color: #1a2a6c;
                margin: 20px 0;
                font-weight: bold;
            }
            
            .confidence {
                color: #666;
                font-size: 1.2em;
            }
            
            .loading {
                display: none;
                width: 50px;
                height: 50px;
                border: 5px solid #f3f3f3;
                border-top: 5px solid #1a2a6c;
                border-radius: 50%;
                margin: 20px auto;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            audio {
                width: 100%;
                margin-top: 20px;
                border-radius: 30px;
            }
            
            .instructions {
                background: rgba(26,42,108,0.1);
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
            }
            
            .instructions h3 {
                color: #1a2a6c;
                margin-bottom: 15px;
            }
            
            .instructions ul {
                list-style: none;
            }
            
            .instructions li {
                margin: 10px 0;
                padding-left: 25px;
                position: relative;
                color: #444;
            }
            
            .instructions li:before {
                content: "•";
                color: #1a2a6c;
                font-weight: bold;
                position: absolute;
                left: 0;
            }
            
            .error {
                background: #ffebee;
                color: #b71c1c;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
            }
        </style>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container">
            <h1>Sign Language Detection</h1>
            
            <div class="instructions">
                <h3>Instructions for Best Results:</h3>
                <ul>
                    <li>Position yourself in good lighting</li>
                    <li>Keep your hands clearly visible</li>
                    <li>Make slow, deliberate gestures</li>
                    <li>Complete the full sign motion</li>
                    <li>Record for 1-2 seconds only</li>
                </ul>
            </div>
            
            <div class="tabs">
                <div class="tab active" data-tab="camera">
                    <i class="fas fa-camera"></i> Camera
                </div>
                <div class="tab" data-tab="upload">
                    <i class="fas fa-upload"></i> Upload
                </div>
            </div>
            
            <div id="camera-tab" class="tab-content active">
                <div class="camera-container">
                    <video id="camera-feed" autoplay playsinline></video>
                    <div class="recording-indicator"></div>
                    <div class="timer">00:00</div>
                </div>
                
                <div class="camera-controls">
                    <button id="start-camera" class="control-btn">
                        <i class="fas fa-video"></i> Start Camera
                    </button>
                    <button id="record" class="control-btn" disabled>
                        <i class="fas fa-circle"></i> Start Recording
                    </button>
                    <button id="stop-record" class="control-btn">
                        <i class="fas fa-stop"></i> Stop Recording
                    </button>
                    <button id="process-recording" class="control-btn" disabled>
                        <i class="fas fa-cog"></i> Process
                    </button>
                </div>
            </div>
            
            <div id="upload-tab" class="tab-content">
                <div class="upload-form">
                    <form id="uploadForm">
                        <div class="file-input-container">
                            <label for="video" class="control-btn">
                                <i class="fas fa-file-video"></i> Choose Video File
                            </label>
                            <input type="file" id="video" name="video" accept="video/*" required>
                            <div class="file-name" id="fileName">No file chosen</div>
                        </div>
                        <button type="submit" class="control-btn">
                            <i class="fas fa-cog"></i> Process Video
                        </button>
                    </form>
                </div>
            </div>
            
            <div class="loading" id="loading"></div>
            <div id="result" class="result" style="display: none;"></div>
            <audio id="audio" controls style="display: none;"></audio>
        </div>

        <script>
            let mediaRecorder;
            let recordedChunks = [];
            let isRecording = false;
            let timerInterval;
            let startTime;
            
            const cameraFeed = document.getElementById('camera-feed');
            const startCameraBtn = document.getElementById('start-camera');
            const recordBtn = document.getElementById('record');
            const stopRecordBtn = document.getElementById('stop-record');
            const processBtn = document.getElementById('process-recording');
            const recordingIndicator = document.querySelector('.recording-indicator');
            const timer = document.querySelector('.timer');
            
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    tab.classList.add('active');
                    document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
                });
            });
            
            // Camera handling
            startCameraBtn.addEventListener('click', async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: 640,
                            height: 480,
                            frameRate: { ideal: 30 }
                        }
                    });
                    cameraFeed.srcObject = stream;
                    
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) {
                            recordedChunks.push(e.data);
                        }
                    };
                    
                    mediaRecorder.onstop = () => {
                        stopRecording();
                    };
                    
                    startCameraBtn.disabled = true;
                    recordBtn.disabled = false;
                } catch (err) {
                    console.error('Error accessing camera:', err);
                    alert('Could not access camera. Please ensure you have granted camera permissions.');
                }
            });
            
            function updateTimer() {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const seconds = elapsed % 60;
                const minutes = Math.floor(elapsed / 60);
                timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
            
            recordBtn.addEventListener('click', () => {
                startRecording();
            });
            
            stopRecordBtn.addEventListener('click', () => {
                mediaRecorder.stop();
            });
            
            function startRecording() {
                recordedChunks = [];
                mediaRecorder.start();
                isRecording = true;
                recordBtn.style.display = 'none';
                stopRecordBtn.style.display = 'inline-flex';
                processBtn.disabled = true;
                recordingIndicator.classList.add('active');
                startTime = Date.now();
                timer.classList.add('active');
                timerInterval = setInterval(updateTimer, 1000);
                
                // Auto-stop after 2 seconds
                setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.stop();
                    }
                }, 2000);
            }
            
            function stopRecording() {
                isRecording = false;
                recordBtn.style.display = 'inline-flex';
                stopRecordBtn.style.display = 'none';
                processBtn.disabled = false;
                recordingIndicator.classList.remove('active');
                timer.classList.remove('active');
                clearInterval(timerInterval);
            }
            
            processBtn.addEventListener('click', async () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const formData = new FormData();
                formData.append('video', blob);
                
                processVideo(formData);
            });
            
            // File upload handling
            document.getElementById('video').addEventListener('change', function(e) {
                const fileName = e.target.files[0] ? e.target.files[0].name : "No file chosen";
                document.getElementById('fileName').textContent = fileName;
            });

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const videoFile = document.getElementById('video').files[0];
                if (!videoFile) {
                    alert('Please select a video file');
                    return;
                }

                const formData = new FormData();
                formData.append('video', videoFile);
                
                processVideo(formData);
            });
            
            async function processVideo(formData) {
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                const audio = document.getElementById('audio');

                loading.style.display = 'block';
                result.style.display = 'none';
                audio.style.display = 'none';

                try {
                    const response = await fetch('/process_video', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    loading.style.display = 'none';

                    if (data.success) {
                        result.innerHTML = `
                            <h3>Result:</h3>
                            <div class="prediction">${data.prediction}</div>
                            <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
                        `;
                        result.style.display = 'block';
                        
                        audio.src = `data:audio/mp3;base64,${data.audio}`;
                        audio.style.display = 'block';
                    } else {
                        result.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                        result.style.display = 'block';
                    }
                } catch (error) {
                    loading.style.display = 'none';
                    result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    result.style.display = 'block';
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        video_file = request.files['video']
        temp_path = 'temp_video.mp4'
        video_file.save(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        sequence = []
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.7,  # Increased confidence threshold
            min_tracking_confidence=0.7,
            model_complexity=1  # Better accuracy
        ) as holistic:
            while cap.isOpened() and len(sequence) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (640, 480))  # Larger size for better detection
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                try:
                    # Extract keypoints with better precision
                    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                    
                    keypoints = np.concatenate([
                        pose[:132],
                        lh[:31],
                        rh[:32]
                    ])
                    
                    # Add smoothing for stability
                    if len(sequence) > 0:
                        prev_keypoints = sequence[-1]
                        keypoints = 0.7 * keypoints + 0.3 * prev_keypoints
                        
                    sequence.append(keypoints)
                except:
                    continue
        
        cap.release()
        os.remove(temp_path)
        
        if len(sequence) > 0:
            if len(sequence) < 30:
                sequence.extend([sequence[-1]] * (30 - len(sequence)))
            sequence = sequence[:30]
            input_data = np.array(sequence)
            
            # Get prediction with confidence
            prediction = model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_idx])
            predicted_action = actions[predicted_idx]
            
            # Generate audio
            tts = gTTS(text=predicted_action, lang='en')
            audio_path = 'temp_audio.mp3'
            tts.save(audio_path)
            
            with open(audio_path, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode()
            os.remove(audio_path)
            
            return jsonify({
                'success': True,
                'prediction': predicted_action,
                'confidence': confidence,
                'audio': audio_base64
            })
            
        return jsonify({
            'success': False,
            'error': 'No valid frames detected'
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("Server running on http://localhost:5000")
    app.run(debug=True, port=5000)
