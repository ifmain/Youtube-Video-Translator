import os
from service_libs import *
import time




clener()
creator()

url=input('Enter URL:')
saver(url)

inst_path=None
vocal_path=None
vid_path=None

for a in onlyfiles('temp/audio'):
    if '_Instruments' in a:
        inst_path=f'temp\\audio\\"{a}"'
    else:
        vocal_path=f'temp\\audio\\"{a}"'

for a in onlyfiles('temp/save'):
    if '.mp4' in a:
        vid_path=f'temp/save/"{a}"'


os.system(f'move {inst_path} temp\\A1.wav')
inst_path="temp\\A1.wav"
os.system(f'move {vocal_path} temp\\voc.wav')
vocal_path="temp/voc.wav"


os.system(f'ffmpeg -i {vid_path} -c:v copy -an temp/V.mp4')

outdir = 'temp/out'
timings = split_audio(vocal_path, outdir)

for a in onlyfiles('temp/out'):
    print(a)
    result = pipe(f'temp/out/{a}')
    tts(translate(result['text']))
    speedup(f'temp/out/{a}', 'temp/data/output.mp3',output_path=f'temp/out2/{a}')

silentfile = 'temp/data/silent.mp3'
finalfile = 'temp/A2.wav'
newfragdir = 'temp/out2'

create_silent_audio(vocal_path, silentfile)
merge_fragments(silentfile, finalfile, newfragdir, timings)

os.system('ffmpeg -i temp/V.mp4 -i temp/A1.wav -c:v copy temp/V1.mp4')
os.system('ffmpeg -i temp/A2.wav -c:v copy -af "volume=7dB" temp/A2_BB.wav')

os.system('ffmpeg -i temp/V1.mp4 -i temp/A2_BB.wav -c:v copy -filter_complex "[0:a][1:a] amix=inputs=2:duration=longest [audio_out]" -map 0:v -map "[audio_out]" -y Translated.mp4')
time.sleep(1)
clener()