import os
import torch
from cog import BasePredictor, Input, Path
from datetime import timedelta
import whisperx
from whisper.utils import get_writer
import uuid
import json
import requests
import tempfile
from pytube import YouTube
from urllib.parse import urlparse


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        os.environ['HF_HOME'] = '/src/hf_models'
        os.environ['TORCH_HOME'] = '/src/torch_models'
        self.compute_type = "float16"
        self.language_code = "pt"
        self.result = {}

        self.model = whisperx.load_model('large-v2', self.device, language=self.language_code, compute_type=self.compute_type, download_root="whisper-cache")
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code=self.language_code, device=self.device)

    @staticmethod
    def download_file(url):
        local_filename = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix="_" + local_filename) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        return f.name

    def predict(
        self,
        audio: str = Input(description="Audio url"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        align_output: bool = Input(description="Use if you need word-level timing and not just batched transcription", default=False),
        only_text: bool = Input(description="Set if you only want to return text; otherwise, segment metadata will be returned as well.", default=False),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        # ensure to use your own library or methods

        if isinstance(audio, str) and bool(urlparse(audio).netloc):
            audio = Predictor.download_file(audio)

        self.result['audio_path'] = audio

        """Run a single prediction on the model"""
        with torch.inference_mode():
            whisper_result = self.model.transcribe(audio, batch_size=batch_size, language="pt")
            output_directory = '.'

            # ensure to use your own library or methods
            self.result['whisper'] = whisper_result

            srt_writer = get_writer("srt", output_directory)
            srt_writer(whisper_result, 'test.srt')

            json_writer = get_writer("json", output_directory)
            json_writer(whisper_result, 'test.json')

            if align_output:
                # ensure to use your own library or methods
                self.result['whisper']['segments'] = whisperx.align(segments, self.alignment_model, self.metadata, audio, self.device, return_char_alignments=False)['segments']

            # self.result['whisper']['only_text'] = ''.join([val.text for val in self.result['whisper']['segments']]); 

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return json.dumps(self.result)
