import cv2
import threading
import pyaudio
import audioop
from flask import Flask, Response, render_template_string, request, jsonify
import time
import socket
import webbrowser
from collections import deque
import numpy as np

app = Flask(__name__)

# Global variables for camera and audio
latest_frame = None
frame_lock = threading.Lock()
audio_buffer = deque(maxlen=44100 * 2)  # 2 seconds buffer

# Audio configuration
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.frame_data = {}
        self.lock = threading.Lock()
        
    def add_camera(self, camera_id=0):
        """Add a camera to the manager"""
        with self.lock:
            if camera_id not in self.cameras:
                self.cameras[camera_id] = {
                    'running': False,
                    'thread': None,
                    'camera': None
                }
                self.frame_data[camera_id] = None
                return True
        return False
    
    def start_camera(self, camera_id=0):
        """Start a camera"""
        with self.lock:
            if camera_id in self.cameras and not self.cameras[camera_id]['running']:
                self.cameras[camera_id]['running'] = True
                thread = threading.Thread(
                    target=self._camera_loop, 
                    args=(camera_id,), 
                    daemon=True
                )
                self.cameras[camera_id]['thread'] = thread
                thread.start()
                return True
        return False
    
    def stop_camera(self, camera_id=0):
        """Stop a camera"""
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id]['running'] = False
                if self.cameras[camera_id]['camera']:
                    self.cameras[camera_id]['camera'].release()
                return True
        return False
    
    def _camera_loop(self, camera_id):
        """Camera capture loop"""
        camera = cv2.VideoCapture(camera_id)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        with self.lock:
            self.cameras[camera_id]['camera'] = camera
        
        try:
            while self.cameras[camera_id]['running']:
                success, frame = camera.read()
                if not success:
                    print(f"Camera {camera_id} read failed")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Encode as JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                with self.lock:
                    self.frame_data[camera_id] = frame_bytes
                
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Camera {camera_id} error: {e}")
        finally:
            camera.release()
            with self.lock:
                self.cameras[camera_id]['running'] = False
    
    def _process_frame(self, frame):
        """Process frame with timestamp"""
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, timestamp, (18, 38), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        return frame
    
    def get_frame(self, camera_id):
        """Get latest frame from camera"""
        with self.lock:
            return self.frame_data.get(camera_id)

class AudioManager:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.lock = threading.Lock()
        self.audio_device_index = self._find_camera_microphone()
        
    def _find_camera_microphone(self):
        """Try to find microphone associated with camera"""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        # Look for devices that might be camera microphones
        camera_keywords = ['camera', 'webcam', 'video', 'integrated', 'array']
        
        for i in range(0, num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            device_name = device_info.get('name', '').lower()
            
            # Check if device has input channels and matches camera keywords
            if device_info.get('maxInputChannels', 0) > 0:
                print(f"Audio Device {i}: {device_info['name']} - {device_info['maxInputChannels']} input channels")
                
                # Prefer devices that sound like camera microphones
                if any(keyword in device_name for keyword in camera_keywords):
                    print(f"🎤 Selected camera microphone: {device_info['name']}")
                    return i
        
        # If no camera mic found, use default input device
        default_input = self.audio.get_default_input_device_info()
        print(f"🎤 Using default microphone: {default_input['name']}")
        return None  # None will use default device
    
    def start_recording(self):
        """Start audio recording from camera microphone"""
        with self.lock:
            if self.recording:
                return False
            
            try:
                self.stream = self.audio.open(
                    format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=self.audio_device_index,
                )
                self.recording = True
                print("🎤 Audio recording started")
                return True
            except Exception as e:
                print(f"Audio recording error: {e}")
                # Fallback to default device
                try:
                    self.stream = self.audio.open(
                        format=AUDIO_FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=None
                    )
                    self.recording = True
                    print("🎤 Audio recording started with default device")
                    return True
                except Exception as e2:
                    print(f"Fallback audio recording error: {e2}")
                    return False
    
    def stop_recording(self):
        """Stop audio recording"""
        with self.lock:
            if self.stream and self.recording:
                self.recording = False
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                print("🎤 Audio recording stopped")
                return True
            return False
    
    def read_audio(self):
        """Read audio data"""
        with self.lock:
            if self.recording and self.stream:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    # Normalize audio volume
                    rms = audioop.rms(data, 2)
                    if rms > 1000:
                        data = audioop.mul(data, 2, min(2.0, 3000.0 / max(1, rms)))
                    return data
                except Exception as e:
                    print(f"Audio read error: {e}")
                    return None
            return None
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        self.audio.terminate()

# Initialize managers
camera_manager = CameraManager()
audio_manager = AudioManager()

def get_local_ip():
    """Get local IP address for network access"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "127.0.0.1"

def audio_capture_loop():
    """Continuous audio capture loop"""
    print("🔊 Starting audio capture loop...")
    
    while True:
        try:
            if audio_manager.recording:
                audio_data = audio_manager.read_audio()
                if audio_data:
                    audio_buffer.append(audio_data)
            time.sleep(CHUNK / RATE)
        except Exception as e:
            print(f"Audio capture loop error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """Main page with clean modern design"""
    local_ip = get_local_ip()
    
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VisionStream • Live Camera & Audio</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --secondary: #10b981;
                --danger: #ef4444;
                --dark: #1f2937;
                --darker: #111827;
                --light: #f8fafc;
                --gray: #6b7280;
                --card-bg: rgba(255, 255, 255, 0.05);
                --border: rgba(255, 255, 255, 0.1);
            }}

            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 100%);
                color: var(--light);
                min-height: 100vh;
                line-height: 1.6;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}

            /* Header Styles */
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 40px 20px;
            }}

            .logo {{
                font-size: 3rem;
                margin-bottom: 16px;
            }}

            .header h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 12px;
            }}

            .header p {{
                font-size: 1.1rem;
                color: var(--gray);
                max-width: 600px;
                margin: 0 auto;
            }}

            /* Network Info */
            .network-info {{
                background: var(--card-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 24px;
                margin: 30px 0;
                text-align: center;
            }}

            .network-info h3 {{
                font-size: 1.2rem;
                margin-bottom: 16px;
                color: var(--light);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }}

            .url-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin: 20px 0;
            }}

            .url-item {{
                background: rgba(0, 0, 0, 0.3);
                padding: 16px;
                border-radius: 12px;
                border: 1px solid var(--border);
            }}

            .url-label {{
                font-size: 0.9rem;
                color: var(--gray);
                margin-bottom: 4px;
            }}

            .url-value {{
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 0.95rem;
                color: var(--light);
                word-break: break-all;
            }}

            /* Card Styles */
            .card {{
                background: var(--card-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 24px;
                transition: transform 0.2s ease, border-color 0.2s ease;
            }}

            .card:hover {{
                border-color: rgba(99, 102, 241, 0.3);
                transform: translateY(-2px);
            }}

            .card-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 24px;
            }}

            .card-title {{
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 1.4rem;
                font-weight: 600;
            }}

            .card-icon {{
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
            }}

            /* Video Feed */
            .video-container {{
                position: relative;
                border-radius: 16px;
                overflow: hidden;
                background: #000;
                margin-bottom: 20px;
            }}

            .video-feed {{
                width: 100%;
                max-height: 600px;
                display: block;
                border-radius: 16px;
            }}

            /* Audio Visualizer */
            .audio-visualizer {{
                width: 100%;
                height: 120px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 16px;
                margin: 20px 0;
                overflow: hidden;
                position: relative;
            }}

            .visualizer-bars {{
                display: flex;
                align-items: end;
                justify-content: space-around;
                height: 100%;
                padding: 0 20px;
            }}

            .visualizer-bar {{
                width: 12px;
                background: linear-gradient(to top, var(--primary), var(--secondary));
                border-radius: 6px 6px 0 0;
                transition: height 0.1s ease;
                min-height: 4px;
            }}

            /* Controls */
            .controls {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 12px;
                margin: 24px 0;
            }}

            .btn {{
                padding: 16px 24px;
                border: none;
                border-radius: 12px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                text-decoration: none;
            }}

            .btn-primary {{
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
            }}

            .btn-success {{
                background: linear-gradient(135deg, var(--secondary), #059669);
                color: white;
            }}

            .btn-danger {{
                background: linear-gradient(135deg, var(--danger), #dc2626);
                color: white;
            }}

            .btn-secondary {{
                background: rgba(255, 255, 255, 0.1);
                color: var(--light);
                border: 1px solid var(--border);
            }}

            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }}

            .btn:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }}

            /* Status */
            .status {{
                padding: 16px;
                border-radius: 12px;
                margin: 16px 0;
                text-align: center;
                font-weight: 500;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }}

            .status-active {{
                background: rgba(16, 185, 129, 0.1);
                color: var(--secondary);
                border: 1px solid rgba(16, 185, 129, 0.3);
            }}

            .status-inactive {{
                background: rgba(239, 68, 68, 0.1);
                color: var(--danger);
                border: 1px solid rgba(239, 68, 68, 0.3);
            }}

            .status-info {{
                background: rgba(99, 102, 241, 0.1);
                color: var(--primary);
                border: 1px solid rgba(99, 102, 241, 0.3);
            }}

            /* Stats */
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 16px;
                margin-top: 24px;
            }}

            .stat-item {{
                text-align: center;
                padding: 20px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 12px;
                border: 1px solid var(--border);
            }}

            .stat-value {{
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 4px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}

            .stat-label {{
                font-size: 0.9rem;
                color: var(--gray);
            }}

            /* Responsive */
            @media (max-width: 768px) {{
                .container {{
                    padding: 16px;
                }}

                .header h1 {{
                    font-size: 2rem;
                }}

                .url-grid {{
                    grid-template-columns: 1fr;
                }}

                .controls {{
                    grid-template-columns: 1fr;
                }}

                .card {{
                    padding: 20px;
                }}

                .stats {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}

            /* Loading Animation */
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}

            .loading {{
                animation: pulse 2s infinite;
            }}

            /* Audio Player */
            .audio-player {{
                width: 100%;
                margin: 16px 0;
                border-radius: 12px;
                background: rgba(0, 0, 0, 0.3);
            }}

            /* Connection Status */
            .connection-status {{
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                z-index: 1000;
            }}

            .connected {{
                background: var(--secondary);
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="connection-status connected">
            ● Connected
        </div>

        <div class="container">
            <!-- Header -->
            <div class="header">
                <div class="logo">📹</div>
                <h1>VisionStream</h1>
                <p>Live high-quality video and audio streaming from your camera</p>
            </div>

            <!-- Network Access -->
            <div class="network-info">
                <h3>🌐 Network Access</h3>
                <p>Access your stream from any device on the network</p>
                <div class="url-grid">
                    <div class="url-item">
                        <div class="url-label">Local Access</div>
                        <div class="url-value">http://localhost:5000</div>
                    </div>
                    <div class="url-item">
                        <div class="url-label">Network Access</div>
                        <div class="url-value">http://{local_ip}:5000</div>
                    </div>
                </div>
            </div>

            <!-- Video Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="card-icon">📹</div>
                        <span>Live Video Feed</span>
                    </div>
                </div>

                <div class="video-container">
                    <img src="/video_feed" class="video-feed" alt="Live Camera Feed" 
                         onerror="this.style.display='none'">
                </div>

                <div class="controls">
                    <button class="btn btn-primary" onclick="startCamera()">
                        <span>▶</span> Start Camera
                    </button>
                    <button class="btn btn-danger" onclick="stopCamera()">
                        <span>⏹</span> Stop Camera
                    </button>
                    <button class="btn btn-secondary" onclick="refreshFeed()">
                        <span>🔄</span> Refresh
                    </button>
                </div>

                <div class="status status-info" id="cameraStatus">
                    <span>●</span> Camera ready to start
                </div>
            </div>

            <!-- Audio Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="card-icon">🎤</div>
                        <span>Camera Audio</span>
                    </div>
                </div>

                <audio id="audioPlayer" controls class="audio-player" style="display: none;">
                    Your browser does not support the audio element.
                </audio>

                <div class="audio-visualizer">
                    <div class="visualizer-bars" id="visualizerBars"></div>
                </div>

                <div class="controls">
                    <button class="btn btn-success" onclick="listenToCamera()" id="listenBtn">
                        <span>👂</span> Listen to Camera
                    </button>
                    <button class="btn btn-danger" onclick="stopListening()" id="stopListenBtn" disabled>
                        <span>🔇</span> Stop Listening
                    </button>
                    <button class="btn btn-secondary" onclick="refreshAudio()">
                        <span>🔄</span> Refresh Audio
                    </button>
                </div>

                <div class="status status-info" id="audioStatus">
                    <span>●</span> Ready to listen to camera audio
                </div>

                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value" id="audioLevel">0%</div>
                        <div class="stat-label">Audio Level</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{RATE}Hz</div>
                        <div class="stat-label">Sample Rate</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="bufferSize">0</div>
                        <div class="stat-label">Buffer</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="streamStatus">●</div>
                        <div class="stat-label">Status</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let audioContext;
            let analyser;
            let isListening = false;
            let visualizationBars = 32;

            // Initialize visualizer
            function initVisualizer() {{
                const visualizerBars = document.getElementById('visualizerBars');
                visualizerBars.innerHTML = '';
                for (let i = 0; i < visualizationBars; i++) {{
                    const bar = document.createElement('div');
                    bar.className = 'visualizer-bar';
                    bar.style.height = '4px';
                    visualizerBars.appendChild(bar);
                }}
            }}

            // Update visualization
            function updateVisualization() {{
                if (!analyser || !isListening) return;

                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(dataArray);
                
                const bars = document.querySelectorAll('.visualizer-bar');
                const audioLevel = document.getElementById('audioLevel');
                
                let maxLevel = 0;
                bars.forEach((bar, i) => {{
                    const index = Math.floor(i * dataArray.length / visualizationBars);
                    const value = dataArray[index] || 0;
                    const height = Math.max(4, (value / 255) * 100);
                    bar.style.height = height + 'px';
                    maxLevel = Math.max(maxLevel, value);
                }});

                const levelPercent = Math.round((maxLevel / 255) * 100);
                audioLevel.textContent = levelPercent + '%';
                
                if (isListening) {{
                    requestAnimationFrame(updateVisualization);
                }}
            }}

            // Listen to camera audio
            async function listenToCamera() {{
                try {{
                    const response = await fetch('/control/audio/start', {{ method: 'POST' }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        analyser = audioContext.createAnalyser();
                        analyser.fftSize = 256;

                        const audioPlayer = document.getElementById('audioPlayer');
                        audioPlayer.style.display = 'block';
                        audioPlayer.src = '/audio_feed?t=' + Date.now();
                        
                        audioPlayer.onplay = () => {{
                            const source = audioContext.createMediaElementSource(audioPlayer);
                            source.connect(analyser);
                            analyser.connect(audioContext.destination);
                            initVisualizer();
                            updateVisualization();
                        }};

                        audioPlayer.load();
                        await audioPlayer.play();

                        updateUI('listening', '🎧 Listening to camera audio...', 'status-active');
                        startBufferMonitoring();
                    }} else {{
                        updateUI('error', '❌ ' + data.message, 'status-inactive');
                    }}
                }} catch (error) {{
                    console.error('Error:', error);
                    updateUI('error', '❌ Error starting audio', 'status-inactive');
                }}
            }}

            // Stop listening
            async function stopListening() {{
                const response = await fetch('/control/audio/stop', {{ method: 'POST' }});
                const data = await response.json();
                
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.pause();
                audioPlayer.src = '';
                audioPlayer.style.display = 'none';
                
                initVisualizer();
                document.getElementById('audioLevel').textContent = '0%';
                updateUI('stopped', '🔇 Audio stopped', 'status-inactive');
            }}

            // Update UI state
            function updateUI(state, message, statusClass) {{
                const listenBtn = document.getElementById('listenBtn');
                const stopBtn = document.getElementById('stopListenBtn');
                const audioStatus = document.getElementById('audioStatus');
                const streamStatus = document.getElementById('streamStatus');

                isListening = state === 'listening';
                listenBtn.disabled = isListening;
                stopBtn.disabled = !isListening;
                audioStatus.textContent = message;
                audioStatus.className = `status ${{statusClass}}`;
                streamStatus.textContent = isListening ? '●' : '○';
                streamStatus.style.color = isListening ? '#10b981' : '#6b7280';
            }}

            // Camera controls
            async function startCamera() {{
                const response = await fetch('/control/camera/start', {{ method: 'POST' }});
                const data = await response.json();
                updateCameraStatus(data);
            }}

            async function stopCamera() {{
                const response = await fetch('/control/camera/stop', {{ method: 'POST' }});
                const data = await response.json();
                updateCameraStatus(data);
            }}

            function updateCameraStatus(data) {{
                const status = document.getElementById('cameraStatus');
                status.textContent = data.message;
                status.className = data.success ? 'status status-active' : 'status status-inactive';
            }}

            function refreshFeed() {{
                const video = document.querySelector('.video-feed');
                video.src = '/video_feed?t=' + Date.now();
                document.getElementById('cameraStatus').textContent = '🔄 Video feed refreshed';
                document.getElementById('cameraStatus').className = 'status status-info';
            }}

            function refreshAudio() {{
                if (isListening) {{
                    const audioPlayer = document.getElementById('audioPlayer');
                    audioPlayer.src = '/audio_feed?t=' + Date.now();
                    audioPlayer.load();
                    audioPlayer.play();
                    document.getElementById('audioStatus').textContent = '🔄 Audio feed refreshed';
                    document.getElementById('audioStatus').className = 'status status-info';
                }}
            }}

            // Buffer monitoring
            function startBufferMonitoring() {{
                setInterval(async () => {{
                    const response = await fetch('/audio_info');
                    const data = await response.json();
                    document.getElementById('bufferSize').textContent = data.buffer_size;
                }}, 1000);
            }}

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {{
                initVisualizer();
                startBufferMonitoring();
                
                // Auto-start camera
                startCamera();
            }});

            // Auto-refresh video if needed
            setInterval(() => {{
                const video = document.querySelector('.video-feed');
                if (video && video.naturalWidth === 0) {{
                    refreshFeed();
                }}
            }}, 10000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

# ... (Keep all the backend routes and functions the same as previous version)
@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = camera_manager.get_frame(0)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_feed')
def audio_feed():
    """Audio streaming route"""
    def generate_audio():
        while True:
            if audio_buffer:
                audio_data = audio_buffer.popleft()
                yield audio_data
            else:
                silence = b'\x00' * CHUNK * 2
                yield silence
                time.sleep(CHUNK / RATE)
    
    return Response(generate_audio(), mimetype='audio/wav')

@app.route('/audio_info')
def audio_info():
    """Get audio buffer information"""
    return jsonify({
        'buffer_size': len(audio_buffer),
        'sample_rate': RATE,
        'channels': CHANNELS
    })

@app.route('/control/camera/<action>', methods=['POST'])
def control_camera(action):
    """Camera control endpoint"""
    if action == 'start':
        success = camera_manager.start_camera(0)
        return jsonify({
            'success': success,
            'message': 'Camera started successfully' if success else 'Camera already running or failed to start'
        })
    elif action == 'stop':
        success = camera_manager.stop_camera(0)
        return jsonify({
            'success': success,
            'message': 'Camera stopped successfully' if success else 'Camera not running'
        })

@app.route('/control/audio/<action>', methods=['POST'])
def control_audio(action):
    """Audio control endpoint"""
    if action == 'start':
        success = audio_manager.start_recording()
        return jsonify({
            'success': success,
            'message': 'Audio recording started successfully' if success else 'Audio already running or failed to start'
        })
    elif action == 'stop':
        success = audio_manager.stop_recording()
        audio_buffer.clear()
        return jsonify({
            'success': success,
            'message': 'Audio recording stopped successfully' if success else 'Audio not running'
        })

def open_browser():
    """Open browser automatically"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    camera_manager.add_camera(0)
    camera_manager.start_camera(0)
    
    audio_thread = threading.Thread(target=audio_capture_loop, daemon=True)
    audio_thread.start()
    
    local_ip = get_local_ip()
    
    print("🎥🎤 Starting VisionStream Server...")
    print("=" * 50)
    print(f"📍 Local Access: http://localhost:5000")
    print(f"🌐 Network Access: http://{local_ip}:5000")
    print("=" * 50)
    print("✨ Features:")
    print("   • Modern glassmorphism design")
    print("   • High-quality video streaming")
    print("   • Real-time audio visualization")
    print("   • Camera microphone audio")
    print("   • Mobile-responsive interface")
    print("=" * 50)
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
    finally:
        audio_manager.cleanup()
        print("✅ Server stopped successfully")
