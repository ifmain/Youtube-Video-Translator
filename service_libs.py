import os
os.system('cls||clear')
print('Load libs')

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment, silence
from os.path import isfile, join
from pytube import YouTube
from os import listdir
from gtts import gTTS
import subprocess
import shutil
import torch

os.system('cls||clear')
print('Load models')


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v2"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)


def onlyfiles(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def Download(link):
    print('Download Video')
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download('temp/save')
    except:
        print("An error has occurred")
    print("Download is completed successfully")

def clener():
    try:
        shutil.rmtree('temp')
    except:
        pass

def creator():
    directories = ['temp', 'temp/data', 'temp/out', 'temp/out2']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def saver(url):
    Download(url)

    yt_path = f'temp/save/"{onlyfiles("temp/save")[0]}"'
    yt_aud_path=yt_path.replace("mp4","mp3")
    os.system(f'ffmpeg -i {yt_path} {yt_aud_path}')

    os.system(f'python vocal-remover\inference.py --input {yt_aud_path} --gpu 0 --output_dir temp/audio')






def translate(inp):
    trans_model_path='Helsinki-NLP/opus-mt-en-ru' # Change model in this string to transale on your language
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_path)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_path)
    input_ids = trans_tokenizer(inp, return_tensors="pt").input_ids
    outputs = trans_model.generate(input_ids=input_ids)
    return trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def onlyfiles(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def split_audio(audio_path, outdir, min_silence_len=1000, silence_thresh=-50):
    audio = AudioSegment.from_mp3(audio_path)
    nonsilent_parts = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    timings = []
    for i, (start_i, end_i) in enumerate(nonsilent_parts):
        chunk = audio[start_i:end_i]
        timings.append((start_i, end_i))
        print(f'Фрагмент {i}: начало {start_i // 1000}с, конец {end_i // 1000}с')
        chunk.export(os.path.join(outdir, f'fragment_{start_i // 1000}_{end_i // 1000}.mp3'), format='mp3')
    
    return timings

def create_silent_audio(audio_path, silentfile):
    original_audio = AudioSegment.from_mp3(audio_path)
    silent_audio = AudioSegment.silent(duration=len(original_audio))
    silent_audio.export(silentfile, format='mp3')

def merge_fragments(silentfile, finalfile, fragments_dir):
    final_audio = AudioSegment.from_mp3(silentfile)
    for fragment_file in sorted(os.listdir(fragments_dir)):
        if fragment_file.endswith('.mp3'):
            fragment = AudioSegment.from_mp3(os.path.join(fragments_dir, fragment_file))
            start_time = int(fragment_file.split('_')[1].split('.')[0]) * 1000
            final_audio = final_audio.overlay(fragment, position=start_time)
    final_audio.export(finalfile, format='mp3')

def create_silent_audio(audio_path, silentfile):
    original_audio = AudioSegment.from_mp3(audio_path)
    silent_audio = AudioSegment.silent(duration=len(original_audio))
    silent_audio.export(silentfile, format='mp3')

def merge_fragments(silentfile, finalfile, fragments_dir, timings):
    final_audio = AudioSegment.from_mp3(silentfile)
    for timing in timings:
        fragment_file = f'fragment_{timing[0] // 1000}_{timing[1] // 1000}.mp3'
        fragment_path = os.path.join(fragments_dir, fragment_file)
        if os.path.exists(fragment_path):
            fragment = AudioSegment.from_mp3(fragment_path)
            final_audio = final_audio.overlay(fragment, position=timing[0])

    final_audio.export(finalfile, format='mp3')

def tts(text_to_speech):
    tts = gTTS(text_to_speech, lang='ru')
    tts.save("temp/data/output.mp3")

def speedup(etalon_path, file_to_speedup_path, output_path="out.mp3"):
    # Загрузка аудиофайлов
    etalon = AudioSegment.from_file(etalon_path)
    file_to_speedup = AudioSegment.from_file(file_to_speedup_path)

    # Вычисление коэффициента ускорения
    speedup_factor = 1 / (len(etalon) / len(file_to_speedup))
    if speedup_factor<0.5:
        speedup_factor=0.5
    # Создание команды для ffmpeg
    cmd = [
        "ffmpeg", 
        "-i", file_to_speedup_path, 
        "-filter:a", f"atempo={speedup_factor}",
        "-vn", 
        output_path
    ]

    # Запуск процесса ffmpeg
    subprocess.run(cmd)