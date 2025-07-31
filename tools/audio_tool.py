'''helper functions for handling audio content'''

from llama_index.readers.whisper import WhisperReader
from youtube_transcript_api import YouTubeTranscriptApi


def transcribe_audio_tool(file_path: str) -> str:
    """
    Use this to transcribe audio files (mp3, wav, etc.)
    Args:
    file_path (str): local path of the file to transcribe
    """
    whisper_reader = WhisperReader(model="whisper-1")
    docs = whisper_reader.load_data(file_path)
    return docs[0].text if docs else "[No transcription found]"

def transcribe_youtube(url: str) -> str:
    """Fetch and return transcript from a YouTube video."""
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"[YouTube Transcript Error] {str(e)}"
