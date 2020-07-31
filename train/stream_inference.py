# coding=utf-8
'''
@ Summary: 输入音频文件, 测试模型准确率
@ Update:  

@ file:    stream_inference.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2020/7/29 下午1:42
'''

import sys
import os
sys.path.append('../..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.python.ops import io_ops
from tensorflow.python.ops import gen_audio_ops as audio_ops

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
    frontend_op = None

import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
import scipy.io.wavfile as wav
import scipy.signal
import tensorflow.compat.v1 as tf1

import logging
from pathlib import Path
from argparse import Namespace

from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.layers.modes import Modes
from kws_streaming.models import model_params
from kws_streaming.train import model_flags
from kws_streaming.train import test
from kws_streaming import data

MAX_ABS_INT16 = 32768


def get_wav(wav_file, flags, target_sample_rate=16000, flag=False):
    """ 读取 wav 文件, 返回 input_data 和 采样率"""
    # with tf.io.gfile.GFile(wav_file, 'rb') as file_handle:
    #     samplerate, wave_data = wav.read(file_handle)
    #
    # desired_length = int(
    #     round(float(len(wave_data)) / samplerate * target_sample_rate))
    # wave_data = scipy.signal.resample(wave_data, desired_length)
    #
    # # Normalize short ints to floats in range [-1..1).
    # data = np.array(wave_data, np.float32) / 32768.0
    #
    # if flag:
    #     plt.plot(wave_data)
    # plt.show()
    #
    # # pad input audio with zeros, so that audio len = flags.desired_samples
    # padded_wav = np.pad(wave_data, (0, flags.desired_samples-len(wave_data)),
    #                     'constant')

    wav_loader = io_ops.read_file(wav_file)
    wav_decoder = tf.audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=flags.desired_samples)

    # 在 mfcc 和 micro 模式下, mfcc 获取部分需要手动获取
    if flags.preprocess == 'mfcc':
        # padded_wav = np.expand_dims(padded_wav, axis=-1)  # [16000, 1]
        # Run the spectrogram and MFCC ops to get a 2D audio: Short-time FFTs
        # spectrogram: [channels/batch, frames, fft_feature]
        spectrogram = audio_ops.audio_spectrogram(wav_decoder.audio,
                      window_size=flags.window_size_samples,
                      stride=flags.window_stride_samples,
                      magnitude_squared=flags.fft_magnitude_squared)

        # extract mfcc features from spectrogram by audio_ops.mfcc:
        # 1 Input is spectrogram frames.
        # 2 Weighted spectrogram into bands using a triangular mel filterbank
        # 3 Logarithmic scaling
        # 4 Discrete cosine transform (DCT), return lowest dct_coefficient_count
        # mfcc: [channels/batch, frames, dct_coefficient_count]
        mfcc = audio_ops.mfcc(
            spectrogram=spectrogram,
            sample_rate=flags.sample_rate,
            upper_frequency_limit=flags.mel_upper_edge_hertz,
            lower_frequency_limit=flags.mel_lower_edge_hertz,
            filterbank_channel_count=flags.mel_num_bins,
            dct_coefficient_count=flags.dct_num_features)
        input_data = mfcc
    elif flags.preprocess == 'micro':
        if not frontend_op:
            raise Exception(
                'Micro frontend op is currently not available when running'
                ' TensorFlow directly from Python, you need to build and run'
                ' through Bazel')
        int16_input = tf.cast(
            tf.multiply(wav_decoder.audio, MAX_ABS_INT16), tf.int16)
        # audio_microfrontend does:
        # 1. A slicing window function of raw audio
        # 2. Short-time FFTs
        # 3. Filterbank calculations
        # 4. Noise reduction
        # 5. PCAN Auto Gain Control
        # 6. Logarithmic scaling

        # int16_input dims: [time, channels]
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=flags.sample_rate,
            window_size=flags.window_size_ms,
            window_step=flags.window_stride_ms,
            num_channels=flags.mel_num_bins,
            upper_band_limit=flags.mel_upper_edge_hertz,
            lower_band_limit=flags.mel_lower_edge_hertz,
            out_scale=1,
            out_type=tf.float32)
        # int16_input dims: [frames, num_channels]
        input_data = tf.multiply(micro_frontend, (10.0 / 256.0))
        input_data = tf.expand_dims(input_data, axis=0)
    else:
        # [1, 16000]
        input_data = tf.reshape(wav_decoder.audio, (1, flags.desired_samples))

    return input_data, wav_decoder.sample_rate


def pre_idx_to_word(flags):
    """ prepare index to word"""
    wanted_words = flags.wanted_words.split(',')
    labels = [data.input_data.SILENCE_LABEL, data.input_data.UNKNOWN_WORD_LABEL] + wanted_words \
                if flags.split_data else wanted_words
    index_to_label = dict(zip([i for i in range(len(labels))], labels))
    return index_to_label, labels


def pre_model(train_dir, flags, modes=Modes.NON_STREAM_INFERENCE):
    """ prepare non_stream_model """
    # Prepare model with flag's parameters
    model_non_stream_batch = models.MODELS[flags.model_name](flags)

    # Load model's weights
    weights_name = 'best_weights'
    model_non_stream_batch.load_weights(train_dir/weights_name).expect_partial()
    model_non_stream_batch.summary()

    # convert model to inference mode with batch one
    inference_batch_size = 1
    flags.batch_size = inference_batch_size

    model_non_stream = utils.to_streaming_inference(model_non_stream_batch,
                                            flags, modes)
    # model_non_stream.summary()
    return model_non_stream


def run_inference(input_data, model, FLAG):
    predictions = model.predict(input_data, steps=1)
    predicted_labels = np.argmax(predictions,)
    if index_to_label[predicted_labels] == 'xrxr':
        if FLAG == 4:
            print(f"predicted label: {index_to_label[predicted_labels]}")
            print("=="*15)
            print()
        FLAG += 1
        return FLAG
    return 0


def stream_inference():
    CHUNK = 320  # 20 ms 帧移
    CHANNELS = 1  # 单通道
    RATE = 16000  # 16k 采样率
    FRAMES = 49  # 49 帧
    FORMAT = pyaudio.paInt16
    HEAD = b'RIFF$}\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80' \
           b'>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00}\x00\x00'

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # print(p.get_sample_size(FORMAT))

    q = Queue()
    # init queue
    for _ in range(50):
        q.put(b"\x00" * CHUNK * 2)

    flag, wav_data = 0, 0
    root_path = Path('../local_data/record2')
    if not root_path.exists():  root_path.mkdir()
    try:
        i = int(max(root_path.glob('*.wav')).stem)
    except ValueError:
        i = 0
        print('No wav file here!')

    print("Start recording...")

    while True:
        if flag == 5:
            # save wav
            if not i:  i = 0
            wav_path = root_path / (str(i) + '.wav')
            with wav_path.open('wb') as wf:
                wf = open(str(wav_path), 'wb')
                wf.write(wav_data)
            # init queue
            for _ in range(50):
                q.get()
                q.put(b"\x00" * CHUNK * 2)
            flag, i = 0, i+1
            sys.stdout.flush()
            continue
        data = stream.read(CHUNK)
        # 入队
        q.put(data)
        # 出队
        q.get()

        wav_data = HEAD + b''.join(list(q.queue))
        flag = run_inference(input_data, model)


def np_softmax(logits):
    """ numpy softmax"""
    exp_x = np.exp(logits)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def print_details(wav_file, index_to_label, predicted_labels, predictions,
                  top):
    print(f"file name: {wav_file.name}")
    print(f"predicted label: {index_to_label[predicted_labels]}")
    print(f"scores: {predictions[top[-1]]:.4f}")
    print("=="*15)
    print()


def wav_inference(wav_file, model, flags, index_to_label, wanted_words, count,
                  Threshold=0.8):
    """ 对单个音频文件进行推理, 返回预测标签的索引和分值"""
    input_data, sr = get_wav(str(wav_file), flags)
    assert sr == 16000, print('Sample rate is not 16k!')

    # inference with non stream model
    logits = model.predict(input_data, steps=1)
    logits = tf.squeeze(logits, axis=0)
    predictions = np_softmax(logits)
    predicted_labels = np.argmax(predictions,)
    top = np.argsort(predictions)

    # 强制刷新缓存
    sys.stdout.flush()
    y_true_index = wanted_words.index('xrxr')

    # 判断是不是 xrxr 标签数据
    flag = True if wav_file.parent.name == 'xrxr' else False
    if flag:
        if predicted_labels != y_true_index:
            print_details(wav_file, index_to_label, predicted_labels, predictions, top)
        elif predictions[top[-1]] <= Threshold:
            print_details(wav_file, index_to_label, predicted_labels, predictions, top)
            count += 1
    else:
        if predicted_labels == y_true_index and predictions[top[-1]] > Threshold:
            print_details(wav_file, index_to_label, predicted_labels, predictions, top)
            count += 1
    return predicted_labels, f"{predictions[top[-1]]:.4f}", count


def main(_):
    # 启动动态图机制
    tf1.enable_eager_execution()
    tf1.reset_default_graph()

    # 设置 GPU 按需分配
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    tf1.keras.backend.set_learning_phase(0)

    # Load flags
    model_name = 'dscnn'
    train_dir = Path('../train_model') / model_name
    flags_path = train_dir / 'flags.txt'
    with flags_path.open() as fr:
        flags_txt = fr.read()
    flags = eval(flags_txt)
    flags.data_dir = '../../data'

    # Prepare mapping of index to word
    index_to_label, wanted_words = pre_idx_to_word(flags)

    # prepare model
    model = pre_model(train_dir, flags)

    # Load wav file
    root_path = Path('../../local_data/test_wav/other')

    # 单个文件的测试
    # wav_inference(root_path/'1.wav', model, flags, index_to_label)

    count = 0
    for index, wav_file in enumerate(root_path.glob('*.wav')):
        # parent_dir = wav_file.parent.name
        _, _, count = wav_inference(wav_file, model, flags, index_to_label, wanted_words, count,
                                    0.8)
    print(f"count: {count}")


if __name__ == '__main__':
    # 仅使用 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf1.app.run()


