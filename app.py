import streamlit as st
import os
import gdown
import yt_dlp
import re
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
from sklearn.preprocessing import StandardScaler

# Set Streamlit App Title
st.title("🎵 Audio Emotion Classification")
st.write("Upload an audio file or enter a YouTube URL to analyze the emotion.")

# ✅ Model Setup
MODEL_PATH = "best_Over_model2_anecha_fold2.keras"
GOOGLE_DRIVE_FILE_ID = "1X_Uo31Bzb6Hr322YOzFNsbeWOxPXnYL1"

# ✅ Download Model if Not Available
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model from Google Drive... This may take a while.")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# ✅ Load the Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ✅ Function to Validate YouTube URL
def is_valid_youtube_url(url):
    patterns = [r'(?:v=|v/|embed/|youtu.be/)([a-zA-Z0-9_-]{11})']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)  # Return Video ID
    return None

# ✅ Function to Download YouTube Audio
def download_youtube_audio(url):
    video_id = is_valid_youtube_url(url)
    if not video_id:
        return None, "Invalid YouTube URL format"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'socket_timeout': 30,
        'retries': 3,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            filepath = f"{info['id']}.mp3"
            return filepath, None
    except Exception as e:
        return None, str(e)

# ✅ Function to Convert Audio to WAV
def convert_audio_to_wav(input_path):
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    os.system(f"ffmpeg -y -i {input_path} -ac 1 -ar 48000 {output_path}")
    return output_path if os.path.exists(output_path) else None
def pad_or_truncate(mel_spectrogram, target_shape=(128, 128)):
    """
    Ensures the Mel spectrogram is exactly (128, 128).
    Pads if too small, truncates if too large.
    """
    current_shape = mel_spectrogram.shape
    if current_shape[1] < target_shape[1]:
        pad_width = target_shape[1] - current_shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif current_shape[1] > target_shape[1]:
        mel_spectrogram = mel_spectrogram[:, :target_shape[1]]

    if mel_spectrogram.shape[0] < 128:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 128 - mel_spectrogram.shape[0]), (0, 0)), mode='constant')

    return mel_spectrogram

# ✅ Function to Extract Features
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
        print(f"Loaded WAV file: Shape {y.shape}, Sample rate {sr}")
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
    mel_spec_scaled_flattened = scaler.fit_transform(mel_spec_flattened.reshape(-1, 1)).flatten()
    mel_spec_scaled = mel_spec_scaled_flattened.reshape(target_shape)

    # ✅ Debugging - Print Shape Before Reshaping
    print(f"Mel Spec Shape Before Reshaping: {mel_spec_scaled.shape}")

    # ✅ Ensure correct input shape (Should be 128x128 before adding extra dimensions)
    try:
        mel_spec_scaled = mel_spec_scaled.reshape(1, 128, 128, 1)
        print(f"Mel Spec Shape After Reshaping: {mel_spec_scaled.shape}")
    except ValueError as e:
        print(f"Reshape Error: {e}")
        raise ValueError(f"Reshape failed: {mel_spec_scaled.shape}")

    return mel_spec_scaled

# ✅ Function to Predict Emotion
def predict_emotion(mel_spec):
    try:
        pred = model.predict(mel_spec)
        pred = float(pred)

        emotion_label = "Positive" if pred >= 0.5 else "Negative"
        confidence = pred if pred >= 0.5 else (1 - pred)

        return {
            'primary_emotion': emotion_label,
            'confidence': f'{confidence * 100:.1f}%',
            'distribution': {
                'Positive': max(0, min(100, pred * 100)),
                'Negative': max(0, min(100, (1 - pred) * 100))
            }
        }
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}

# 🔹 Streamlit UI Components
st.sidebar.header("Upload Audio or Enter YouTube URL")

# ✅ YouTube URL Input
youtube_url = st.sidebar.text_input("Enter a YouTube URL")

if st.sidebar.button("Download & Analyze YouTube Audio"):
    if youtube_url:
        filepath, error = download_youtube_audio(youtube_url)
        if error:
            st.error(f"Error downloading YouTube audio: {error}")
        else:
            with st.spinner("Extracting features..."):
                mel_spec = extract_features(filepath)
            with st.spinner("Predicting emotion..."):
                result = predict_emotion(mel_spec)
            st.write(result)
            os.remove(filepath)  # Cleanup

# ✅ File Upload Option
uploaded_file = st.sidebar.file_uploader("Upload an Audio File", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    filepath = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Extracting features..."):
        mel_spec = extract_features(filepath)

    with st.spinner("Predicting emotion..."):
        result = predict_emotion(mel_spec)

    st.write(result)
    os.remove(filepath)  # Cleanup

st.sidebar.write("Developed with ❤️ using Streamlit 🚀")
