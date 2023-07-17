import os
import torch
from cog import BasePredictor, Input, Path
from datetime import timedelta
import whisperx
import uuid
import json
import requests
import tempfile
from urllib.parse import urlparse


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        os.environ['HF_HOME'] = '/src/hf_models'
        os.environ['TORCH_HOME'] = '/src/torch_models'
        self.compute_type = "float16"
        self.language_code = "pt"

        self.model = whisperx.load_model('small', self.device, language=self.language_code, compute_type=self.compute_type, download_root="whisper-cache")
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code=self.language_code, device=self.device)


    def download_youtube_video(url):
        youtube = YouTube(url)
        video = youtube.streams.first()
        video_file = video.download()

        return video_file

    def extract_audio_from_video(video_file):
        video_clip = VideoFileClip(video_file)
        audio_file = video_file.replace(".mp4", ".mp3")  # Replace .mp4 extension with .mp3
        video_clip.audio.write_audiofile(audio_file)

        return audio_file

    def download_file(url):
        local_filename = url.split('/')[-1]

        if "youtube" in url:
            video_file = download_youtube_video(url)
            audio_file = extract_audio_from_video(video_file)
            os.remove(video_file)  # Delete the video file after extracting audio

            with tempfile.NamedTemporaryFile(delete=False, suffix="_" + local_filename) as f:
                with open(audio_file, 'rb') as audio:
                    f.write(audio.read())
            os.remove(audio_file)  # Delete the local audio file after writing to temp file

            return f.name

        else:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix="_" + local_filename) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            return f.name

    @staticmethod
    def create_srt_from_segments(segments):
        for segment in segments:
            startTime = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
            endTime = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
            text = segment['text']
            segmentId = segment['id'] + 1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

            srtFilename = os.path.join(r".", f"{uuid.uuid4()}.srt")

            with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)

        return srtFilename

    def predict(
        self,
        audio: str = Input(description="Audio / Youtube url"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        align_output: bool = Input(description="Use if you need word-level timing and not just batched transcription", default=False),
        only_text: bool = Input(description="Set if you only want to return text; otherwise, segment metadata will be returned as well.", default=False),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        # ensure to use your own library or methods
        
        result = {}

        if isinstance(audio, str) and bool(urlparse(audio).netloc):
            audio = download_file(audio)

        result['audio_path'] = audio

        """Run a single prediction on the model"""
        with torch.inference_mode():
            # ensure to use your own library or methods
            result = self.model.transcribe(audio, batch_size=batch_size, language="pt")

            segments = result['segments']
            result['srt_filename'] = Predictor.create_srt_from_segments(segments)

            if align_output:
                # ensure to use your own library or methods
                result = self.align(segments, self.alignment_model, self.metadata, str(audio), self.device, return_char_alignments=False)

            if only_text:
                return ''.join([val.text for val in segments])

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return json.dumps(result)
