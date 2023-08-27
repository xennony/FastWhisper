from faster_whisper import WhisperModel

model_size = "medium"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

segments, info = model.transcribe("YOUR_FILE.mp3/.mp4 or others", beam_size=5)

print()
print("\033[35mDetected language '%s' with probability %f\033[0m" % (info.language, info.language_probability))
print()

for segment in segments:
            start_time = seconds_to_hms(segment.start)
            end_time = seconds_to_hms(segment.end)
            print("\033[94m[%s -> %s]\033[0m %s" % (start_time, end_time, segment.text))

print()
print("\033[35mReady!\033[0m")
