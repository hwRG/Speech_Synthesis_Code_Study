import pandas as pd
import numpy as np
import os, librosa, re, glob, scipy
from tqdm import tqdm
from util.hparams import *
from util.text import text_to_sequence


# Text 에 대한 내용
text_dir = './archive/transcript.v.1.4.txt'
# 잡다한 것들 string 저장
filters = '([.,!?])'

# 경로, 대사, 실제 발음, 자음모음 분리, 시간, 영어발음
metadata = pd.read_csv(text_dir, dtype='object', sep='|', header=None)

#경로 및 자모 분리 텍스트 가져오기
wav_dir = metadata[0].values
text = metadata[3].values

out_dir = './data'
# exist_ok=True는 경로가 없을 때 때 자동 생성해주기
os.makedirs(out_dir, exist_ok=True)
# text, mel, dec, spec 각자 폴더 만들어 저장경로 확보
os.makedirs(out_dir + '/text', exist_ok=True)
os.makedirs(out_dir + '/mel', exist_ok=True)
os.makedirs(out_dir + '/dec', exist_ok=True)
os.makedirs(out_dir + '/spec', exist_ok=True)

# text
print('Load Text')
text_len = []

# 순서 파악을 위해 enumerate을 사용, 진행률 확인용 tqdm
for idx, s in enumerate(tqdm(text)):
    # re의 정규 표현식을 활용해 문자열에 변형
    # replace와 비슷하게 보면 될 듯
    sentence = re.sub(re.compile(filters), '', s)
    # 텍스트를 각자 자음 모음에 매핑된 숫자에 맞게 변환
    sentence = text_to_sequence(sentence)
    text_len.append(len(sentence))

    # 각 텍스트별 이름 순서 설정
    text_name = 'kss-text-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/text', text_name), sentence, allow_pickle=False)
np.save(os.path.join(out_dir + '/text_len.npy'), np.array(text_len))
print('Text Done')

# audio
print('Load Audio')
mel_len_list = []
for idx, fn in enumerate(tqdm(wav_dir)):
    file_dir = './archive/kss/'+ fn
    # sample rate는 22050으로 설정
    wav, _ = librosa.load(file_dir, sr=sample_rate)
    # 오디오의 앞과 뒤 공백을 적당히 삭제
    wav, _ = librosa.effects.trim(wav)

    # 1차원으로 필터링 진행?
    wav = scipy.signal.lfilter([1, -preemphasis], [1], wav)

    # wav에 Short Term Fourier Transform 진행 파라미터는 1024 256 1024 대입
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft = np.abs(stft)

    # Mel filter bank 생성 → 선형 매트릭스 생성되어 FFT bin을 mel-frequency bin에 투영 가능
    mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_dim)
    # filter와 stft된 wav를 행렬곱 진행한걸 mel_spectrogram으로 함
    mel_spec = np.dot(mel_filter, stft)

    # 스펙트로그램에 mel_spec과 0.00001 중에 큰 것을 선정해 로그 취하고 20 곱하기
    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec))
    # 이 부분 논문 확인 필요
    stft = 20 * np.log10(np.maximum(1e-5, stft))

    # 0.00000001과 1 사이를 벗어나는 수치를 각각 최소 최대로 맞춰줌
    mel_spec = np.clip((mel_spec - ref_db + max_db) / max_db, 1e-8, 1)
    stft = np.clip((stft - ref_db + max_db) / max_db, 1e-8, 1)

    # 전치 후 타입 float 변경
    mel_spec = mel_spec.T.astype(np.float32)
    stft = stft.T.astype(np.float32)

    # mel_spec의 내용의 길이를 담는다
    mel_len_list.append([mel_spec.shape[0], idx])


    # padding
    # reduction 5에 따른 패딩 진행
    remainder = mel_spec.shape[0] % reduction
    if remainder != 0: 
        # n - f + 1을 위한 padding
        mel_spec = np.pad(mel_spec, [[0, reduction - remainder], [0, 0]], mode='constant')
        stft = np.pad(stft, [[0, reduction - remainder], [0, 0]], mode='constant')

    mel_name = 'kss-mel-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/mel', mel_name), mel_spec, allow_pickle=False)

    stft_name = 'kss-spec-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/spec', stft_name), stft, allow_pickle=False)


    # Decoder Input
    # mel_dim(80)과 reduction(5)의 행에 따라 reshape 진행
    # 만약 패딩이 필요한데 진행하지 않으면 에러
    mel_spec = mel_spec.reshape((-1, mel_dim * reduction))

    # mel_spec의 가장 상단을 0으로 깔고, 마지막 열을 제외한 모든 것을 이어 붙임
    dec_input = np.concatenate((np.zeros_like(mel_spec[:1, :]), mel_spec[:-1, :]), axis=0)
    # mel_dim(80) 이후부터 값들을 dec_input에 저장
    dec_input = dec_input[:, -mel_dim:]

    dec_name = 'kss-dec-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/dec', dec_name), dec_input, allow_pickle=False)

mel_len = sorted(mel_len_list)
np.save(os.path.join(out_dir + '/mel_len.npy'), np.array(mel_len))
print('Audio Done')
