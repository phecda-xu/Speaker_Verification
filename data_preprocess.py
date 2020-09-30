import os
import librosa
import numpy as np
from configuration import parser


def save_spectrogram_tisv(args):
    """
    prepare aishell_1 data
    :param args:
    :return:
    """
    print("start text independent utterance feature extraction")

    total_speaker_num = 0
    for data_set in os.listdir(args.audio_path):
        os.makedirs(os.path.join(args.feature_path, data_set), exist_ok=True)  # make folder to save test file
        speaker_num = len(os.listdir(os.path.join(args.audio_path, data_set)))
        total_speaker_num += speaker_num
        print("{} speaker number: {}".format(data_set, speaker_num))
    print("total speaker number : %d" % total_speaker_num)
    for data_set in os.listdir(args.audio_path):
        for i, folder in enumerate(os.listdir(os.path.join(args.audio_path, data_set))):
            speaker_path = os.path.join(args.audio_path, data_set, folder)  # path of each speaker
            print("%dth speaker processing..." % i)
            utterances_spec = []
            for utter_name in os.listdir(speaker_path):
                utter_path = os.path.join(speaker_path, utter_name)  # path of each utterance
                utter, sr = librosa.core.load(utter_path, args.sr)  # load utterance audio
                S = librosa.core.stft(y=utter, n_fft=args.nfft,
                                      win_length=int(args.window * sr), hop_length=int(args.hop * sr))
                S = np.abs(S) ** 2
                mel_basis = librosa.filters.mel(sr=args.sr, n_fft=args.nfft, n_mels=40)
                S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                if S.shape[1] > args.tisv_frame:

                    utterances_spec.append(S[:, :args.tisv_frame])  # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -args.tisv_frame:])  # last 180 frames of partial utterance
                else:
                    continue

            utterances_spec = np.array(utterances_spec)
            print(utterances_spec.shape)
            np.save(os.path.join(args.feature_path, data_set, "speaker%d.npy" % i), utterances_spec)


if __name__ == "__main__":
    args = parser.parse_args()
    save_spectrogram_tisv(args)
