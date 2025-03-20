# Hvisketiske

Hvisketiske is a Python-based project designed specifically to convert the Hviske-v2 model to the CTranslate2 format, enabling its use with Faster-Whisper for optimized performance.

## Features

- **Model Conversion**: Convert the Hviske-v2 model to Faster-Whisper format for efficient ASR.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hvisketiske
   ```

2. Install dependencies using PDM:
   ```bash
   pdm install
   ```

## Usage

Run the `distil_hvisker.py` script to convert the Hviske-v2 model to Faster-Whisper format:
```bash
python distil_hvisker.py
```

## Dependencies

- Python 3.10
- ctranslate2 >= 4.5.0
- transformers >= 4.49.0
- torch >= 2.6.0
- faster-whisper >= 0.9.0

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Author

Mathias Oliver Valdbjørn Rønnelund  
Email: mathiasoliverjorgensen@hotmail.com