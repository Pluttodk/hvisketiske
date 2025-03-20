from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time
import pandas as pd
from faster_whisper import WhisperModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import numpy as np
from datasets import load_dataset
import jiwer

@dataclass
class TranscriptionResult:
    """Class to store transcription results and metrics"""
    text: str
    time_taken: float
    wer: float
    cer: float

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "syvai/hviske-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
hviskev2_pipe = pipe


from faster_whisper import WhisperModel
import numpy as np
output_dir = Path(f"faster_hviskev2")
# Check if CUDA is available
use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA/GPU is not available, falling back to CPU. This will be much slower.")
    device = "cpu"
    compute_type = "int8"  # Using int8 on CPU for better performance
else:
    device = "cuda"
    compute_type = "float16"

# Try to load the model with the selected device
try:
    fast_model = WhisperModel(str(output_dir / "ct2_model"), device=device, compute_type=compute_type)
    print(f"Model loaded successfully using {device}")
except (ValueError, RuntimeError, ImportError) as e:
    print(f"Error loading model with {device}: {e}")
    print("Falling back to CPU")
    device = "cpu"
    compute_type = "int8"  # Using int8 on CPU for better performance
    fast_model = WhisperModel(str(output_dir / "ct2_model"), device=device, compute_type=compute_type)
    print("Model loaded successfully using CPU")

fast_model_turbo = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device=device,
    compute_type=compute_type
)

# Test transcription with the sample
print(f"Transcribing using {device}...")

def hviskev2(audio):
    result = hviskev2_pipe(audio)
    return result["text"]

def distilhviskev2(audio):

    return fast_model.transcribe(audio, language="da")

def turbov3(audio):
    return fast_model_turbo.transcribe(audio, language="da")
    
class ModelEvaluator:
    """Class to evaluate and compare ASR models"""
    def __init__(self):
        # Initialize faster-whisper model (distilled)
        self.distilled_model = WhisperModel(
            str(Path("faster_hviskev2/ct2_model")), 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
        
        # Initialize basic whisper model
        model_name = "syvai/hviske-v2"
        self.basic_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.basic_model = self.basic_model.to("cuda")

        self.turbo_model = WhisperModel(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
    
    def transcribe_distilled(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe using distilled model and return text and time taken"""
        start_time = time.time()
        segments, _ = distilhviskev2(audio_path)
        end_time = time.time()
        
        # Combine all segments
        full_text = " ".join([segment.text for segment in segments])
        return full_text, end_time - start_time
    
    def transcribe_basic(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe using basic model and return text and time taken"""
        
        start_time = time.time()
        transcriptions = hviskev2(audio_path)
        end_time = time.time()
        
        return transcriptions, end_time - start_time

    def transcribe_turbo(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe using turbo model and return text and time taken"""
        start_time = time.time()
        segments, _ = turbov3(audio_path)
        end_time = time.time()
        full_text = " ".join([segment.text for segment in segments])
        return full_text, end_time - start_time
    
    def calculate_metrics(self, hypothesis: str, reference: str) -> Tuple[float, float]:
        """Calculate WER and CER"""
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return wer, cer
    
    def evaluate_sample(self, audio_path: str, reference_text: str) -> Dict[str, TranscriptionResult]:
        """Evaluate all models on a single audio sample"""
        results = {}
        
        # Evaluate models
        for model_name, transcribe_func in [
            ("distilled", self.transcribe_distilled),
            ("basic", self.transcribe_basic),
            ("turbo", self.transcribe_turbo)
        ]:
            text, time_taken = transcribe_func(audio_path)
            wer, cer = self.calculate_metrics(text, reference_text)
            results[model_name] = TranscriptionResult(
                text=text,
                time_taken=time_taken,
                wer=wer,
                cer=cer
            )
        
        return results

def main():
    """Main function to run evaluation"""
    # Load 10 samples from Coral dataset
    dataset = load_dataset("alexandrainst/coral", split="test")
    evaluator = ModelEvaluator()
    
    # Results storage for DataFrame
    results_data = []
    
    print(f"A total samples of {len(dataset)}")
    # Process first 10 samples
    for i, sample in enumerate(dataset):
        print(f"\nProcessing sample {i+1}...")
        
        # Save audio to temporary file
        audio_path = f"temp_audio_{i}.wav"
        import soundfile as sf
        audio_data = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        
        # Save as WAV file
        sf.write(audio_path, audio_data, sampling_rate)
        
        # Evaluate
        results = evaluator.evaluate_sample(audio_path, sample["text"])
        
        # Store results in data list
        for model_type, result in results.items():
            results_data.append({
                "sample_id": i,
                "model": model_type,
                "reference": sample["text"],
                "transcription": result.text,
                "wer": result.wer,
                "cer": result.cer,
                "time": result.time_taken
            })
            
        # Clean up
        Path(audio_path).unlink()
        
        # Print sample results
        print(f"\nSample {i+1} Results:")
        print(f"Reference: {sample['text']}")
        for model_type, result in results.items():
            print(f"\n{model_type.capitalize()} Model:")
            print(f"Transcription: {result.text}")
            print(f"Time: {result.time_taken:.2f}s")
            print(f"WER: {result.wer:.4f}")
            print(f"CER: {result.cer:.4f}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)
    csv_path = Path("model_evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Print average results grouped by model
    print("\nAverage Results:")
    summary = df.groupby("model").agg({
        "wer": ["mean", "std"],
        "cer": ["mean", "std"],
        "time": ["mean", "std"]
    })
    print(summary)
    
    # Save the summary to a file
    summary_csv_path = Path("model_evaluation_summary.csv")
    summary.to_csv(summary_csv_path)
    print(f"\nSummary saved to {summary_csv_path}")

if __name__ == "__main__":
    main()