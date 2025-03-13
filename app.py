from flask import Flask, render_template, request, jsonify
import os
import yt_dlp
import re
import librosa
import numpy as np
import tensorflow as tf
import subprocess
import soundfile as sf
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the modelsimport gdown
import os
import tensorflow as tf

model_path = "best_Over_model2_anecha_fold2.keras"
drive_file_id = "1X_Uo31Bzb6Hr322YOzFNsbeWOxPXnYL1"
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)

# Load the model
model_fold2 = tf.keras.models.load_model(model_path)
print("Model loaded successfully")


# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_youtube_url(url):
    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:v=|v/|embed/|youtu.be/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)  # Return the video ID
    return None

def download_youtube_audio(url):
    try:
        # Extract video ID
        video_id = is_valid_youtube_url(url)
        if not video_id:
            return None, "Invalid YouTube URL format"

        print(f"Processing video ID: {video_id}")
        
        # Configure yt-dlp with timeout and retries
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'socket_timeout': 30,  # Timeout after 30 seconds
            'retries': 3,  # Retry 3 times if download fails
        }
        
        clean_url = f'https://www.youtube.com/watch?v={video_id}'
        print(f"Using cleaned URL: {clean_url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(clean_url, download=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{info['id']}.mp3")
                print(f"Download completed successfully to: {filepath}")
                return filepath, None
            except Exception as e:
                print(f"Download error: {str(e)}")
                return None, f"Download error: Video might be unavailable or restricted"
                
    except Exception as e:
        print(f"YouTube error: {str(e)}")
        return None, f"Error accessing video: Please try a different video"
import os
import subprocess

def convert_audio_to_wav(input_path):
    """
    Converts any audio file to WAV format using FFmpeg.
    Automatically generates the output WAV path.
    """
    try:
        input_path = os.path.normpath(input_path)
        output_path = input_path.rsplit(".", 1)[0] + ".wav"  # Replace extension with .wav
        
        print(f"Converting audio: {input_path} → {output_path}")
        
        if not os.path.isfile(input_path):
            print(f"Source file not found: {input_path}")
            return None
            
        command = ['ffmpeg', '-y', '-i', input_path, '-ac', '1', '-ar', '48000', output_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        if os.path.isfile(output_path):
            print(f"Conversion successful: {output_path}")
            return output_path
        else:
            print(f"Output file not created: {output_path}")
            return None
            
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        return None
    
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
import soundfile as sf
import os

def pad_or_truncate(mel_spectrogram, target_shape=(128, 128)):
    """
    Pads or truncates a Mel spectrogram to a fixed shape.
    """
    current_shape = mel_spectrogram.shape
    if current_shape[1] < target_shape[1]:
        pad_width = target_shape[1] - current_shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif current_shape[1] > target_shape[1]:
        mel_spectrogram = mel_spectrogram[:, :target_shape[1]]
    
    return mel_spectrogram

def reshape_and_stack(mel_list, target_shape=(128, 128)):
    """
    Reshapes, pads/truncates, and stacks Mel spectrograms.
    """
    reshaped_mels = [pad_or_truncate(mel, target_shape) for mel in mel_list]
    return np.stack(reshaped_mels)

def extract_features(audio_file, target_shape=(128, 128)):
    """
    Extracts a Mel Spectrogram from an entire audio file.
    Applies padding, reshaping, and standardization.
    """

    # ✅ Convert to WAV if needed
    if not audio_file.endswith('.wav'):
        converted_audio = convert_audio_to_wav(audio_file)
        if converted_audio is None:
            raise ValueError("Audio conversion failed")
    else:
        converted_audio = audio_file

    # ✅ Load the WAV file safely
    try:
        y, sr = sf.read(converted_audio)
        print(f"Successfully loaded WAV file with shape: {y.shape}, Sample rate: {sr}")
    except Exception as e:
        if os.path.exists(converted_audio):
            os.remove(converted_audio)  # Cleanup on error
        raise ValueError(f"Failed to load WAV file: {e}")

    # ✅ Delete converted file if it was originally non-WAV
    if audio_file != converted_audio:
        os.remove(converted_audio)

    # ✅ Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=512,
                                              window='hamming', hop_length=256, n_mels=128, fmax=sr/2)
    mel_spec = librosa.power_to_db(mel_spec)

    # ✅ Apply padding/truncation
    mel_spec = pad_or_truncate(mel_spec, target_shape)

    # ✅ Standardization (same as training)
    mel_spec_flattened = mel_spec.reshape(-1)  # Flatten
    scaler = StandardScaler()
    mel_spec_scaled_flattened = scaler.fit_transform(mel_spec_flattened.reshape(-1, 1)).flatten()  # Standardize
    mel_spec_scaled = mel_spec_scaled_flattened.reshape(target_shape)  # Reshape back

    # ✅ Ensure correct input shape
    mel_spec_scaled = mel_spec_scaled.reshape(1, 128, 128, 1)

    print(f"Extracted & standardized features from audio file.")
    return mel_spec_scaled

def predict_emotion(mel_spec):
    if model_fold2 is None:
        return {'error': 'Model not loaded'}

    try:
        # Ensure prediction output is a Python float
        pred = model_fold2.predict(mel_spec)  # NumPy array
        pred = float(pred)  # Convert to Python float

        print(f"Model prediction: {pred}")

        # Convert prediction to emotion label
        emotion_label = "Positive" if pred >= 0.5 else "Negative"
        confidence = pred if pred >= 0.5 else (1 - pred)

        return {
            'primary_emotion': emotion_label,
            'confidence': f'{confidence * 100:.1f}%',  # This now works correctly
            'distribution': {
                'Positive': max(0, min(100, pred * 100)),
                'Negative': max(0, min(100, (1 - pred) * 100))
            }
        }
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received upload request")
    
    # Handle YouTube URL
    youtube_url = request.form.get('youtube_url')
    if youtube_url:
        print(f"Processing YouTube URL: {youtube_url}")
        if not is_valid_youtube_url(youtube_url):
            print("Invalid YouTube URL format")
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        filepath, error = download_youtube_audio(youtube_url)
        if error:
            print(f"YouTube download error: {error}")
            return jsonify({'error': f'Error downloading YouTube audio: {error}'}), 400
        
        # Extract features and predict
        try:
            print("Extracting features from YouTube audio")
            mel_spec = extract_features(filepath)
            result = predict_emotion(mel_spec)
            
            # Clean up downloaded file
            os.remove(filepath)
            
            if 'error' in result:
                print(f"Prediction error: {result['error']}")
                return jsonify({'error': result['error']}), 400
                
            return jsonify(result)
        except Exception as e:
            print(f"Processing error: {str(e)}")
            print(f"Error type: {type(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 400

    # Handle file upload
    print("Checking for file upload")
    if 'audio' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    print(f"Processing file: {file.filename}")
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Extract features and predict
            print("Extracting features from uploaded file")
            mel_spec = extract_features(filepath)
            result = predict_emotion(mel_spec)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if 'error' in result:
                print(f"Prediction error: {result['error']}")
                return jsonify({'error': result['error']}), 400
                
            return jsonify(result)
        except Exception as e:
            print(f"Processing error: {str(e)}")
            print(f"Error type: {type(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 400
    
    print("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
