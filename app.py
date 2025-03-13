from flask import Flask, render_template, request, jsonify
import os
import yt_dlp
import re
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the models
print("Attempting to load models...")
try:
    print("Current working directory:", os.getcwd())
    print("Checking if model files exist:")
    print("model_2.keras exists:", os.path.exists('model_2.keras'))
    
    # 
    model_fold2 = tf.keras.models.load_model('model_2.keras')
    print("Model fold 2 loaded successfully")

except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(f"Error type: {type(e)}")
    model_fold2 = None

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

def extract_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, duration=30)  # Limit to 30 seconds
    
    # Extract Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec)
    
    # Ensure consistent shape for mel spectrogram
    target_length = 128  # Model expects (None, 128, 128, 1)
    if mel_spec.shape[1] < target_length:
        mel_spec = np.pad(mel_spec, ((0,0), (0, target_length-mel_spec.shape[1])))
    else:
        mel_spec = mel_spec[:, :target_length]
    
    # Reshape for the model - should be (1, 128, 128, 1)
    mel_spec = mel_spec.reshape(1, 128, 128, 1)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Ensure consistent shape for MFCC
    target_length_mfcc = 204  # This should match your model's expected input
    if mfcc.shape[1] < target_length_mfcc:
        mfcc = np.pad(mfcc, ((0,0), (0, target_length_mfcc-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :target_length_mfcc]
    
    mfcc = mfcc.T  # Transpose to match shape (204, 13)
    mfcc = np.expand_dims(mfcc, 0)  # Add batch dimension
    
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"MFCC shape: {mfcc.shape}")
    
    return mel_spec, mfcc

def predict_emotion(mel_spec, mfcc):
    # 
    if model_fold2 is None or model_fold3 is None:
        return {
            'error': 'Models not loaded'
        }
    
    try:
        # Make predictions with both models
        pred2 = model_fold2.predict([mel_spec, mfcc])[0]
        pred3 = model_fold3.predict([mel_spec, mfcc])[0]
        
        print(f"Model 2 prediction: {pred2}")
        print(f"Model 3 prediction: {pred3}")
        
        # Average the predictions
        prediction = (float(pred2[0]) + float(pred3[0])) / 2
        print(f"Average prediction: {prediction}")
        
        # Convert prediction to emotion
        emotion_label = "Positive" if prediction >= 0.5 else "Negative"
        confidence = prediction if prediction >= 0.5 else (1 - prediction)
        
        # Ensure predictions are within 0-1 range
        positive_score = max(0, min(100, prediction * 100))
        negative_score = max(0, min(100, (1 - prediction) * 100))
        
        print(f"Final scores - Positive: {positive_score}%, Negative: {negative_score}%")
        
        return {
            'primary_emotion': emotion_label,
            'confidence': f'{confidence * 100:.1f}%',
            'distribution': {
                'Positive': positive_score,
                'Negative': negative_score
            }
        }
    except Exception as e:
        print(f"Prediction error details: {str(e)}")
        return {
            'error': f'Prediction error: {str(e)}'
        }

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
            mel_spec, mfcc = extract_features(filepath)
            result = predict_emotion(mel_spec, mfcc)
            
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
            mel_spec, mfcc = extract_features(filepath)
            result = predict_emotion(mel_spec, mfcc)
            
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