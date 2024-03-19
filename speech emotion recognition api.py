from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

import librosa
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = load_model('speech_cnn_model_new.h5')
print(model.summary())
# Define emotions
emotions = ["Angry", "Calm", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# Define the input shape expected by the model
n_timesteps = 162  # Define the number of timesteps
n_features = 1    # Define the number of features (e.g., for MFCC)

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def preprocess_audio(audio_path, n_timesteps, n_features):
    # Load the audio sample
    # audio_sample, sample_rate = librosa.load(audio_path, sr=None)
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)

    # # Extract features (e.g., MFCC)
    # mfcc_features = librosa.feature.mfcc(y=audio_sample, sr=sample_rate, n_mfcc=n_features, hop_length=512).T
    # # Pad or truncate to the desired number of timesteps
    # if len(mfcc_features) < n_timesteps:
    #     mfcc_features = np.pad(mfcc_features, ((0, n_timesteps - len(mfcc_features)), (0, 0)), mode='constant')
    # elif len(mfcc_features) > n_timesteps:
    #     mfcc_features = mfcc_features[:n_timesteps, :]
    # # Add a channel dimension
    # mfcc_features = mfcc_features[..., np.newaxis]
    # Reshape to match the expected input shape of the model
    # return mfcc_features.reshape(-1, n_timesteps, n_features, 1)
    return result

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if audio_file:
        preprocessed_audio = preprocess_audio(audio_file, n_timesteps, n_features)
        preprocessed_audio = np.expand_dims(preprocessed_audio, axis=-1)
        preprocessed_audio = np.expand_dims(preprocessed_audio, axis=1)
        preprocessed_audio = np.expand_dims(preprocessed_audio, axis=0)
        print(preprocessed_audio.shape)
        predicted_label = model.predict(preprocessed_audio)
        predicted_emotions = []
        predicted_class = np.argmax(predicted_label)
        print("Predicted Class Index:", predicted_class)
        print("Predicted Label propability:", predicted_label[0][predicted_class])
        print("Predicted class:", emotions[predicted_class])

        return jsonify({"Predicted Label:": emotions[predicted_class] })

if __name__ == '__main__':
    app.run(debug=True)
