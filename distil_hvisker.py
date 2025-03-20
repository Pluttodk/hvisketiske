import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pathlib import Path
import shutil
import json

# Step 1: Download the hvisker-v2 model from Hugging Face
model_name = "syvai/hviske-v2"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Step 2: Create directory for the faster-whisper model
output_dir = Path(f"faster_hviskev2")
output_dir.mkdir(exist_ok=True)

# Step 3: Export the model in the CTranslate2 format (used by faster-whisper)
# Create necessary directories
model_dir = output_dir / "model"
model_dir.mkdir(exist_ok=True)

# Save the encoder
encoder = model.get_encoder()
encoder_path = model_dir / "encoder.pt"
torch.save(encoder.state_dict(), encoder_path)

# Save the decoder
decoder = model.get_decoder()
decoder_path = model_dir / "decoder.pt"
torch.save(decoder.state_dict(), decoder_path)

# # Save the projection layer
# proj_path = model_dir / "proj.pt"
# torch.save(model.proj.state_dict(), proj_path)

# Save the tokenizer configuration
tokenizer_config = {
    "vocab_size": model.config.vocab_size,
    "bos_token_id": model.config.bos_token_id,
    "eos_token_id": model.config.eos_token_id,
    "pad_token_id": model.config.pad_token_id,
}
with open(output_dir / "tokenizer.json", "w") as f:
    json.dump(tokenizer_config, f)

# Save the model config
model_config = {
    "model_type": "whisper",
    "architectures": ["WhisperForConditionalGeneration"],
    "encoder_layers": model.config.encoder_layers,
    "decoder_layers": model.config.decoder_layers,
    "d_model": model.config.d_model,
    "hidden_size": model.config.hidden_size,
    "num_attention_heads": model.config.num_attention_heads,
}
with open(output_dir / "config.json", "w") as f:
    json.dump(model_config, f)

# Save the vocabulary and merges files
# Instead of using the model name, let's save the tokenizer files directly
vocab_file = output_dir / "vocab.json"
merges_file = output_dir / "merges.txt"
processor.tokenizer.save_vocabulary(str(output_dir))

# Save the processor's feature extractor configuration
with open(output_dir / "preprocessor_config.json", "w") as f:
    json.dump(processor.feature_extractor.to_dict(), f)

# Step 4: Convert to CTranslate2 format using the command line tool to ensure files are copied
ct2_model_dir = str(output_dir / "ct2_model")
os.system(f"ct2-transformers-converter --model {model_name} --output_dir {ct2_model_dir} "
          f"--quantization float16 --force --copy_files tokenizer.json preprocessor_config.json")

# Alternatively, if you prefer to keep using the converter API:
# from ctranslate2.converters import TransformersConverter
# converter = TransformersConverter(model_name)
# converter.convert(
#     output_dir=str(output_dir / "ct2_model"),
#     quantization="float16",
#     force=True
# )
# 
# # Then manually copy the necessary files
# shutil.copy(output_dir / "tokenizer.json", output_dir / "ct2_model" / "tokenizer.json")
# shutil.copy(output_dir / "preprocessor_config.json", output_dir / "ct2_model" / "preprocessor_config.json")

print(f"Model successfully converted to faster-whisper format at {output_dir}")

# Step 5: Use the converted model with faster-whisper
from faster_whisper import WhisperModel
from datasets import load_dataset
import numpy as np

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
    model = WhisperModel(str(output_dir / "ct2_model"), device=device, compute_type=compute_type)
    print(f"Model loaded successfully using {device}")
except (ValueError, RuntimeError, ImportError) as e:
    print(f"Error loading model with {device}: {e}")
    print("Falling back to CPU")
    device = "cpu"
    compute_type = "int8"  # Using int8 on CPU for better performance
    model = WhisperModel(str(output_dir / "ct2_model"), device=device, compute_type=compute_type)
    print("Model loaded successfully using CPU")



# Test transcription with the sample
print(f"Transcribing using {device}...")
segments, info = model.transcribe("coral_wav_files/speakerspe_01fc2b156c7fe429f1b72bd3be5ad3c3_recordingrec_0ac716227244a1178f8f3db2d9ae6249_amagermål.wav", language="da")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")