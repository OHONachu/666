import sys
from faster_whisper import WhisperModel

audio_path = r"C:\Users\ohona\OneDrive\Desktop\666\audio0917.wav"

def fmt(ts):
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
except Exception as e:
    print(f"CUDA failed ({e}), falling back to CPU int8", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    audio_path,
    language="zh",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
print(f"Detected language: {info.language} (p={info.language_probability:.2f}), duration={info.duration:.1f}s", flush=True)

lines = []
for seg in segments:
    line = f"[{fmt(seg.start)} - {fmt(seg.end)}] {seg.text.strip()}"
    lines.append(line)
    print(line, flush=True)

out_txt = r"C:\Users\ohona\OneDrive\Desktop\666\audio0917_transcript.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"WROTE {out_txt} ({len(lines)} lines)", flush=True)
