# coding=utf-8
'''
@ Summary: 输入音频文件, 测试模型准确率
@ Update:  增加模型误唤醒率和唤醒率散点图

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
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
except ImportError:
    frontend_op = None

import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
import scipy.io.wavfile as wav
import scipy.signal
from sklearn.metrics import accuracy_score, confusion_matrix, \
            precision_score, recall_score, f1_score
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
    # model_non_stream_batch.summary()

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


def print_details(wav_file, index_to_label, predicted_labels, predictions):
    print(f"file name: {wav_file.name}")
    print(f"predicted label: {index_to_label[predicted_labels]}")
    print(f"scores: {predictions[predicted_labels]:.4f}")
    print("=="*15)
    print()


def wav_inference(wav_file, model, flags, index_to_label, wanted_words,
                  Threshold=0):
    """ 对单个音频文件进行推理, 返回预测标签的索引和分值"""
    input_data, sr = get_wav(str(wav_file), flags)
    assert sr == 16000, print('Sample rate is not 16k!')

    # inference with non stream model
    logits = model.predict(input_data, steps=1)
    logits = tf.squeeze(logits, axis=0)
    predictions = np_softmax(logits)
    predicted_labels = np.argmax(predictions,)
    scores = predictions[predicted_labels]

    # 强制刷新缓存
    sys.stdout.flush()

    y_true_index = wanted_words.index('xrxr')

    # model 预测的标签值
    y_pred = index_to_label[predicted_labels]

    # 判断是不是 xrxr 标签数据
    if wav_file.parent.name == 'xrxr':
        y_true = 1
        if Threshold and scores < Threshold:
            y_pred = index_to_label[0]
    else:
        y_true = 0
        if Threshold and (predicted_labels == y_true_index and scores < Threshold):
            # print_details(wav_file, index_to_label, predicted_labels, predictions)
            y_pred = index_to_label[0]
    return y_true, y_pred, f"{scores*100:.2f}"


def cal_matrix(train_dir, root_path, model, flags, index_to_label, wanted_words,
               Threshold=0):
    """ 多个文件进行推理, 并计算相应的混淆矩阵 / 准确率 / 唤醒率 / 误唤醒 / 漏唤醒"""
    y_key = ['y_true_list', 'y_pred_list', 'y_score_list']
    y_dict = dict(zip(y_key, [list() for _ in range(3)]))
    for index, wav_file in enumerate(root_path.glob('*/*.wav')):
        y_true, y_pred, y_scores = wav_inference(wav_file, model,
                     flags, index_to_label, wanted_words, Threshold)

        if y_pred == 'xrxr':
            y_dict['y_pred_list'].append(1)
        else:
            y_dict['y_pred_list'].append(0)
        y_dict['y_true_list'].append(y_true)
        y_dict['y_score_list'].append(y_scores)

    matrix = confusion_matrix(y_dict['y_true_list'], y_dict['y_pred_list'], labels=[1, 0])
    tn, fp, fn, tp = confusion_matrix(y_dict['y_true_list'], y_dict['y_pred_list']).ravel()
    acc = accuracy_score(y_dict['y_true_list'], y_dict['y_pred_list'])
    precision = precision_score(y_dict['y_true_list'], y_dict['y_pred_list'])
    recall = recall_score(y_dict['y_true_list'], y_dict['y_pred_list'])
    f1 = f1_score(y_dict['y_true_list'], y_dict['y_pred_list'])
    false_alarm_rate = fp / (fp + tn)  # 误唤醒: FN / (Total Negative)

    # save txt
    model_test_file = train_dir.parent / ('model_threshold_' + str(int(Threshold*100)) + '.txt')
    with model_test_file.open('a') as fw:
        fw.write(f'The model is: {train_dir.name}\n')
        fw.write("Confusion Matrix is:\n{}\n".format(matrix))
        fw.write("The accuracy is: {:.2f}%.\n".format(acc * 100))
        # fw.write(f"The wake rate is: {recall*100:.2f}%.")
        fw.write(f"The false wake rate is: {false_alarm_rate*100:.2f}%.\n")
        fw.write(f"The wake rate is: {recall*100:.2f}%.\n")
        fw.write(f"The f1 score is: {f1*100:.2f}%.\n\n")
        fw.write("==============================\n\n")
    return f'{recall*100:.2f}', f'{false_alarm_rate*100:.2f}'


def plot_far(x_axis, y_axis, train_dir, model_test_png,
             marker, color, threshold=[0]):
    """ 画模型唤醒率-误唤醒率散点图 """
    if len(threshold) == 1:
        model_test_png = model_test_png.parent / 'model_threshold_0.png'

    plt.title(f'Model Performance Test (Threshold={threshold})')
    plt.xlabel('False wake rate')
    plt.ylabel('Wake rate')
    for i in range(len(train_dir)):
        for j in range(len(threshold)):
            if j == 0:
                plt.scatter(x_axis[j][i], y_axis[j][i], s=80, c=color[i],
                            marker=marker[i], label=train_dir[i].name)
            else:
                plt.scatter(x_axis[j][i], y_axis[j][i], s=80, c=color[i],
                            marker=marker[i])
            if len(threshold) > 1:
                plt.annotate(s=f"{threshold[j]}",
                             xytext=(x_axis[j][i], y_axis[j][i]+0.5),
                             xy=(x_axis[j][i], y_axis[j][i]),)
    plt.legend(loc='best')  # show 每个数据对应的标签
    plt.savefig(model_test_png)
    plt.show()


def plot_acc(train_dir, marker, color):
    acc, model_size = list(), list()
    for path in train_dir:
        # 读取准确率
        print(path.name)
        acc_file = path / 'tflite_non_stream' / 'tflite_non_stream_model_accuracy.txt'
        with acc_file.open() as fr:
            test_acc = fr.read().split()[0]
            acc.append(round(float(test_acc), 2))
        # 读取tilite 文件大小
        keras_model = path / 'non_stream' / 'my_model.h5'
        model_size.append((keras_model.stat().st_size)/(10**5))
    for i in range(len(acc)):
        plt.scatter(model_size[i], acc[i], s=80, c=color[i],
                    marker=marker[i], label=train_dir[i].name)
    plt.title('Model size and accuracy')
    plt.xlabel('Model Size kb (*10^5)')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    print(acc)
    print(model_size)

    plt.savefig(train_dir[0].parent / 'size_acc.png')
    plt.show()


def main(_):
    threshold = [0, 0.8, 0.9]
    test_root_path = Path('../../local_data/test_wav')
    train_root_path = Path('../train_model')
    model_test_png = train_root_path / ('model_threshold.png')

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
    train_dir = [path for path in train_root_path.iterdir() if path.is_dir()]
    x_axis, y_axis = list(), list()  # 分别储存误唤醒率和唤醒率
    train_dir = list(filter(lambda x: x.name == 'att_mh_rnn', train_dir))

    for thre in threshold:
        for model_path in train_dir:
            flags_path = model_path / 'flags.txt'
            with flags_path.open() as fr:
                flags_txt = fr.read()
            flags = eval(flags_txt)
            flags.data_dir = '../../data'

            # Prepare mapping of index to word
            index_to_label, wanted_words = pre_idx_to_word(flags)

            # prepare model
            model = pre_model(model_path, flags)

            # 单个文件的测试
            # wav_inference(test_root_path/'1.wav', model, flags, index_to_label)

            # 多个文件进行推理
            print(f"{model_path.name} model inferece start...")
            recall, far = cal_matrix(model_path, test_root_path,
                        model, flags, index_to_label, wanted_words, thre)
            x_axis.append(float(far))
            y_axis.append(float(recall))
            print(f"{model_path.name} model inferece start done!")

    x_axis = np.array(x_axis).reshape((len(threshold), -1))
    y_axis = np.array(y_axis).reshape((len(threshold), -1))

    # 3.07 3.07 2.45 98.48 98.48 97.98
    # scatter
    x_axis = [[4.91, 5.52, 30.06, 3.07, 14.11, 45.4, 16.56, 3.07, 21.47],
              [3.68, 3.07, 11.66, 3.07, 8.59, 19.02, 6.13, 0.0, 17.79],
              [3.07, 2.45, 4.29, 2.45,  5.52, 8.59, 4.91, 0.0, 15.95]]
    y_axis = [[98.48, 95.45, 100.0, 98.48, 99.49, 98.99, 100.0, 98.48, 99.49],
              [98.48, 93.94, 98.48, 98.48, 98.99, 91.41, 96.46, 96.97, 98.99],
              [98.48, 93.43, 96.97, 97.98, 98.48, 75.76, 93.94, 94.44, 98.99]]

    # 散点的形状和颜色
    marker = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
              '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D',
              'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7,
              8, 9, 10, 11]
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'y',
             'b', 'c', 'g', 'k', 'm', 'r', 'y']

    # plot_far(x_axis, y_axis, train_dir, model_test_png,
    #          marker, color, threshold)
    # plot_acc(train_dir, marker, color)




if __name__ == '__main__':
    # 仅使用 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf1.app.run()


