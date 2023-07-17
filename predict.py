import os
import torch
from cog import BasePredictor, Input, Path
from datetime import timedelta
import uuid
import json

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        os.environ['HF_HOME'] = '/src/hf_models'
        os.environ['TORCH_HOME'] = '/src/torch_models'
        self.compute_type = "float16"
        self.language_code = "pt"

        self.model = self.load_model(model_name, self.device, language=self.language_code, compute_type=self.compute_type)
        self.alignment_model, self.metadata = self.load_align_model(language_code=self.language_code, device=self.device)


    @staticmethod
    def create_srt_from_segments(segments):
        for segment in segments:
            startTime = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
            endTime = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
            text = segment['text']
            segmentId = segment['id'] + 1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"

            srtFilename = os.path.join(r".", f"{uuid.uuid4()}.srt")

            with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)

        return srtFilename

    def predict(
        self,
        model_name: str = Input(description='large-v2'),
        audio: Path = Input(description="Audio file"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        align_output: bool = Input(description="Use if you need word-level timing and not just batched transcription", default=False),
        only_text: bool = Input(description="Set if you only want to return text; otherwise, segment metadata will be returned as well.", default=False),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        # ensure to use your own library or methods
        

        """Run a single prediction on the model"""
        with torch.inference_mode():
            # ensure to use your own library or methods
            result = self.model.transcribe(str(audio), batch_size=batch_size, language=self.language_code)

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
