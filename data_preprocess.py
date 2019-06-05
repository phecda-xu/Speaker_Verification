import os
import librosa
import numpy as np
from configuration import get_config

config = get_config()   # get arguments from parser


def save_spectrogram_tisv(audio_path):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num= (total_speaker_num//10)*8            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth speaker processing..."%i)
        utterances_spec = []
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
            S = librosa.core.stft(y=utter, n_fft=config.nfft,
                                  win_length=int(config.window * sr), hop_length=int(config.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

            utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
            utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:                                         # save spectrogram as numpy file
            np.save(os.path.join(config.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(config.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    audio_path = r'/run/user/1001/gvfs/smb-share:server=fs.lm,share=share/AIbot数据整理_10'  # utterance dataset
    save_spectrogram_tisv(audio_path)