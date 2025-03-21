from faster_whisper import WhisperModel

model = WhisperModel("pluttodk/hviske-tiske")

segments, info = model.transcribe("audio.mp3")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))