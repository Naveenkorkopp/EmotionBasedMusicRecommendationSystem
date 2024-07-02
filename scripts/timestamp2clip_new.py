import os
import json
from pydub import AudioSegment

# Load timestamps
with open('../timestamps.json', 'r') as f:
    timestamps = json.load(f)

# Directory for segmented audio clips
segmented_audio_dir = "../audios/seg"
os.makedirs(segmented_audio_dir, exist_ok=True)

# Function to extract clips
def extract_clips(youtube_id, segments, raw_audio_dir, segmented_audio_dir):
    audio_path = os.path.join(raw_audio_dir, f"{youtube_id}.mp3")
    audio = AudioSegment.from_mp3(audio_path)

    for i, segment in enumerate(segments):
        start = segment['start'] * 1000  # pydub works in milliseconds
        end = segment['end'] * 1000
        clip = audio[start:end]
        clip_path = os.path.join(segmented_audio_dir, f"{youtube_id}_{i}.mp3")
        clip.export(clip_path, format="mp3")


raw_audio_dir = '../audios/raw'

# Process each YouTube ID
for youtube_id, segments in timestamps.items():
    extract_clips(youtube_id, segments, raw_audio_dir, segmented_audio_dir)

