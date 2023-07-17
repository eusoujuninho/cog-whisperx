# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json
from datetime import timedelta
import uuid


compute_type="float16"
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model("large-v2", self.device, language="en", compute_type=compute_type)
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)


    def create_srt_from_segments(segments):
        for segment in segments:
            startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
            text = segment['text']
            segmentId = segment['id']+1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"

            srtFilename = os.path.join(r".", f"{uuid.uuid4()}.srt")
                 
            with open(srtFilename, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)

    return srtFilename


    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        language: Path = Input(description = "Audio language code", default="pt"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        align_output: bool = Input(description="Use if you need word-level timing and not just batched transcription", default=False),
        only_text: bool = Input(description="Set if you only want to return text; otherwise, segment metadata will be returned as well.", default=False),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            result = self.model.transcribe(str(audio), batch_size=batch_size, language=language) 
            # result is dict w/keys ['segments', 'language']
            # segments is a list of dicts,each dict has {'text': <text>, 'start': <start_time_msec>, 'end': <end_time_msec> }
            
            segments = result['segments']
            result['srt_filename'] = create_srt_from_segments(segments)
            
            if align_output:
                # NOTE - the "only_text" flag makes no sense with this flag, but we'll do it anyway
                result = whisperx.align(segments, self.alignment_model, self.metadata, str(audio), self.device, return_char_alignments=False)
                # dict w/keys ['segments', 'word_segments']
                # aligned_result['word_segments'] = list[dict], each dict contains {'word': <word>, 'start': <start_time_msec>, 'end': <end_time_msec>, 'score': probability}
                #   it is also sorted
                # aligned_result['segments'] - same as result segments, but w/a ['words'] segment which contains timing information above. 
                # return_char_alignments adds in character level alignments. it is: too many. 
            if only_text:
                return ''.join([val.text for val in segments])
            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        return json.dumps(result)

