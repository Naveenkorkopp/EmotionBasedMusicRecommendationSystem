import os
from yt_dlp import YoutubeDL

# Define the directory to save raw audio files
raw_audio_dir = "../audios/raw1"
os.makedirs(raw_audio_dir, exist_ok=True)

# Read YouTube URLs from the song list file
with open('../songs_lists/song_lists.txt', 'r') as f:
    youtube_urls = f.readlines()

# Define yt-dlp options for downloading audio in mp3 format
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': os.path.join(raw_audio_dir, '%(id)s.%(ext)s'),
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

# Download audio from each YouTube URL
with YoutubeDL(ydl_opts) as ydl:
    for url in youtube_urls:
        try:
            ydl.download([url.strip()])
        except Exception as e:
            print(e)
