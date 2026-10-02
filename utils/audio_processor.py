import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url : str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': r'C:\Users\saiff\OneDrive\Documents\ffmpeg-2026-08-20-git-7d77562d2a-full_build\bin',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file_path = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        return audio_file_path
     

def convert_audio_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_length_ms: int = 600000) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunks = []
    for i,start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def process_input(url: str) -> list:
    if url.startswith("http://") or url.startswith("https://"):
        print(f"Detected YouTube URL: {url}")
        audio_file_path = download_youtube_audio(url)
        wav_path = convert_audio_to_wav(audio_file_path)
    else:
        print("detected local file path. Converting to wav...")
        wav_path = convert_audio_to_wav(url)

    print(f"chunking audio: {wav_path}")
    chunks = chunk_audio(wav_path)
    print(f"Audio chunked into {len(chunks)} parts.")
    return chunks

