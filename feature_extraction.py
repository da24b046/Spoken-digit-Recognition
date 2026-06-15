import os
import shutil
import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  
    n_fft = 512
    hop_length = 160  

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13,n_fft=n_fft,hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    log_energy = np.log(rms + 1e-10)  

    T = mfcc.shape[1]

    if T < 9:
        width = T if T % 2 == 1 else T - 1
        width = max(width, 3)
        delta = librosa.feature.delta(mfcc, width=width)

    else:
        delta = librosa.feature.delta(mfcc)

    log_energy = log_energy[:, :T]
    features = np.vstack([mfcc, log_energy, delta]) # 13 MFCC + 1 log-energy + 13 delta  = 27 features

    return features.T  


def train_test_split():

    audio_folder = "recordings"
    test_folder = "test_recordings"
    val_folder = 'val_recordings'

    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder,exist_ok=True)

    audio_data = []

    for file in os.listdir(audio_folder):
 
        file_path = os.path.join(audio_folder, file)
        features = extract_features(file_path)
        parts = file.split("_")
        label = int(parts[0])
        speaker_id = int(parts[1].replace("speaker", ""))

        if 1 <= speaker_id <= 5:
            audio_data.append((label, features))
            if speaker_id == 4:
                path = os.path.join(val_folder,file)
                shutil.copy(file_path,path)
        elif speaker_id == 6:
            path = os.path.join(test_folder, file)
            shutil.copy(file_path, path)

    return audio_data


if __name__ == '__main__':
    train_test_split()