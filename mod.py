
import pickle
import numpy as np
import librosa

with open('music_genre_model.pkl', 'rb') as file:
    model_dict = pickle.load(file)

clf = model_dict['clf']
knn = model_dict['knn']
clf_svm = model_dict['clf_svm']
clf_dt = model_dict['clf_dt']
clf_rf = model_dict['clf_rf']
scaler = model_dict['scaler']
lookup_genre_name = model_dict['lookup_genre_name']

def metadata(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    metadata_dict = {
        'tempo': tempo,
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spec_centroid),
        'spectral_bandwidth': np.mean(spec_bw),
        'rolloff': np.mean(spec_rolloff),
        'zero_crossing_rates': np.mean(zero_crossing)
    }

    for i in range(1, 21):
        metadata_dict.update({'mfcc'+str(i): np.mean(mfcc[i-1])})


    return list(metadata_dict.values())


from collections import Counter

def predict_genre(file_path):

    predictions = []

    a = metadata(file_path)
    for model in [clf, knn, clf_svm, clf_dt, clf_rf]:
      d1 = np.array([x if np.isscalar(x) else np.mean(x) for x in a])
      data1 = scaler.transform([d1])
      genre_prediction = model.predict(data1)
      predictions.append(lookup_genre_name[genre_prediction[0]])

    final_prediction = Counter(predictions).most_common(1)[0][0]
    print("The song is predicted as:", final_prediction)
    return final_prediction

