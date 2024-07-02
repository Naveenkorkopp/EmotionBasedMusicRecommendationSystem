
import os
import json
from datetime import datetime



def clip(input_name, st, dur, output_name):

    os.system("ffmpeg -i \
        {} \
        -ss {} \
        -t {} \
        {}".format(
            input_name, 
                    st, 
                   dur, 
            output_name
        ))

def calculate_dur(st, ed):
    _st = datetime.strptime(st, "%H:%M:%S")
    _ed = datetime.strptime(ed, "%H:%M:%S")

    dur = str(_ed - _st)


    return st, dur
    

def main():

    timestamp_ROOT = '../timestamps.json'
    audio_ROOT = "../audios/raw1/"
    output_ROOT = "../audios/seg/"

    if not os.path.exists(output_ROOT):
        os.mkdir(output_ROOT)
  


    with open(timestamp_ROOT, 'r') as f:
        stamps = json.load(f)        
        
        
    for key, value in stamps.items():
        input_name = os.path.join(audio_ROOT, key + '.mp3')
        if not os.path.exists(input_name):
            print('audio file not exist:', input_name)
            continue
    
        for i, piece in enumerate(value):
            output_name = os.path.join(output_ROOT, "Q" + piece[2] + "_" + key + "_" + str(i) + '.mp3')
        
            st, dur = calculate_dur(piece[0], piece[1])
            clip(input_name, st, dur, output_name)
            
        
            
if __name__ == "__main__":
    main()
    print("===== Finished all files.=====")
