# -*- encoding: utf-8 -*-

from rapid_paraformer import RapidParaformer
from rapid_paraformer.rapid_punc import PuncParaformer
from rapid_paraformer.rapid_vad import RapidVad
import moviepy.editor as mp
import time
from concurrent.futures import ThreadPoolExecutor
vad_model = RapidVad()
paraformer = RapidParaformer()
punc = PuncParaformer()

#统计时间的装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"function name: {func.__name__}")
        result = func(*args, **kwargs)
        end = time.time() - start
        print(f"cost time: {end}")
        return result
    return wrapper
# 音频时长
@timeit
def get_audio_duration(wav_path):
    import wave
    f = wave.open(wav_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    duration = nframes / framerate
    #转成00:00:00格式
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    duration = "%02d:%02d:%02d" % (h, m, s)
    return duration

#读取音频
from joblib import Parallel, delayed
@timeit
def load_audiro(wav_path):
    '''
    加载音频
    :param wav_path: 音频路径
    :return:
    '''
    #如果wav是mp4格式，需要先转换为wav格式，然后再加载

    # print('加载音频')
    y, sr = librosa.load(wav_path, sr=16000) # 加载音频文件并解码为音频信号数组
    # wav_list = [y[i:i + 16000 * 30] for i in range(0, len(y), 16000 * 30)]
    wav_list = (y[i:i + 16000 * 30] for i in range(0, len(y), 16000 * 30))
    # wav_list = Parallel(n_jobs=-1)(
    #     delayed(lambda i: y[i:i + 16000 * 30])(i)
    #     for i in range(0, len(y), 16000 * 30))
    # print('切割音频完成')
    return  wav_list

def split_string(text, length):
    """
    将字符串按照指定长度拆分并返回拆分后的列表
    """
    result = []
    start = 0
    while start < len(text):
        end = start + length
        result.append(text[start:end])
        start = end
    return result

def text_process(text):
    #句号，问号后面自动换行
    text = text.replace('。', '。\n')
    text = text.replace('？', '？\n')
    return text


import librosa
import threading

@timeit
def load_and_cut_audio(wav_path, num_threads=4, chunk_size=30):
    y, sr = librosa.load(wav_path, sr=16000)
    n_samples = len(y)
    chunk_size_samples = 16000 * chunk_size
    num_chunks = n_samples // chunk_size_samples + (n_samples % chunk_size_samples > 0)
    chunk_indices = [(i * chunk_size_samples, min(n_samples, (i + 1) * chunk_size_samples)) for i in range(num_chunks)]

    results = [None] * num_chunks

    def load_and_cut_thread(start_index, end_index, result_list, index):
        result_list[index] = y[start_index:end_index]

    threads = []
    for i, (start_index, end_index) in enumerate(chunk_indices):
        t = threading.Thread(target=load_and_cut_thread, args=(start_index, end_index, results, i))
        threads.append(t)
        t.start()

        if i % num_threads == num_threads - 1:
            for thread in threads[i - num_threads + 1:i + 1]:
                thread.join()

    for thread in threads[(num_chunks - 1) // num_threads * num_threads:]:
        thread.join()

    return results



@timeit
def vad(vad_model, wav_path):
    return vad_model(wav_path)

from multiprocessing import Pool
from functools import partial

def asr_single(wav):
    try:
        result_text = paraformer(wav)[0][0]
    except:
        result_text = ''
    return result_text
@timeit
def asr(wav_path):
    wav_list = load_audiro(wav_path)
    # wav_list = load_and_cut_audio(wav_path)
    # pool = Pool()
    # result = pool.map(partial(asr_single), wav_list)
    # pool.close()
    # pool.join()
    with ThreadPoolExecutor() as executor:
        result = executor.map(partial(asr_single), wav_list)

    return result

if __name__ == '__main__':
    wave_path = r'C:\Users\ADMINI~1\AppData\Local\Temp\gradio\d5e738ea910657f76c96e6fbfb74f7de8c6fdb11\11.mp3'
    # wave_path = r'E:\10分钟.wav'
    if wave_path.endswith('.mp4'):
        wav_path = wave_path.replace('.mp4', '.wav')
        clip = mp.VideoFileClip(wave_path)
        clip.audio.write_audiofile(wav_path,fps = 22050,bitrate='64k')  # 将剪辑对象的音频部分写入音频文件
        print('mp4转wav完成')
        print(wav_path)
        print(clip.duration)

    #音频时长
    # duration = get_audio_duration(wave_path)
    # print(f"音频时长：{duration}")
    # vad
    # vad_result = vad(vad_model, row_path)
    # print(f"vad结果：{vad_result}")

    #asr
    asr_result = asr(wave_path)
    print('asr完成')
    print(asr_result)

    # print(f"asr结果：{asr_result}")
    #标点
    new_text = punc(''.join(asr_result))
    prossed_text = text_process(new_text[0])
    print(f"标点结果：{prossed_text}")
    #将识别结果写入txt，名称为音频名称
    # with open(f'{wave_path.replace(".mp4", "")}.txt', 'w') as f:
    #     f.write(prossed_text)




