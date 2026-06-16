import os
import json
import subprocess
import whisper
import cv2

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (first time only)...")
        _whisper_model = whisper.load_model("base")
        print("Whisper model loaded!")
    return _whisper_model

MAX_CLIPS = 12
MIN_CLIP_DURATION = 20
MAX_CLIP_DURATION = 60
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())

def transcribe_video(video_path):
    print("Transcribing video with local Whisper...")
    audio_path = video_path.replace(os.path.splitext(video_path)[1], "_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1",
        audio_path
    ], capture_output=True)
    model = get_whisper_model()
    result = model.transcribe(audio_path, verbose=False)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return result.get("segments", [])

def score_segment(text):
    score = 0.0
    viral_keywords = [
        "secret", "never told", "truth", "shocking", "amazing", "incredible",
        "most people", "mistake", "tip", "hack", "strategy", "how to", "why",
        "learn", "change", "life", "money", "success", "story", "actually",
        "honest", "real", "proven", "biggest", "best", "worst", "finally",
        "warning", "stop", "start", "never", "always", "everyone", "everything"
    ]
    text_lower = text.lower()
    for kw in viral_keywords:
        if kw in text_lower:
            score += 1.5
    words = text.split()
    score += min(len(words) / 10, 3.0)
    if "?" in text:
        score += 2.0
    if "!" in text:
        score += 1.0
    return score

def find_best_clips(segments, video_duration):
    if not segments:
        return split_evenly(video_duration)
    for seg in segments:
        seg["score"] = score_segment(seg.get("text", ""))
    candidates = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = start
        text_accumulator = ""
        total_score = 0.0
        j = i
        while j < len(segments) and (end - start) < MAX_CLIP_DURATION:
            end = segments[j]["end"]
            text_accumulator += " " + segments[j].get("text", "")
            total_score += segments[j]["score"]
            j += 1
            clip_duration = end - start
            if clip_duration >= MIN_CLIP_DURATION:
                candidates.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "score": total_score / max(clip_duration, 1),
                    "text": text_accumulator.strip()
                })
    if not candidates:
        return split_evenly(video_duration)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for candidate in candidates:
        overlap = False
        for sel in selected:
            if not (candidate["end"] <= sel["start"] or candidate["start"] >= sel["end"]):
                overlap = True
                break
        if not overlap:
            selected.append(candidate)
        if len(selected) >= MAX_CLIPS:
            break
    selected.sort(key=lambda x: x["start"])
    return selected

def split_evenly(duration):
    chunk = min(45, duration / MAX_CLIPS)
    clips = []
    t = 0
    while t + chunk <= duration and len(clips) < MAX_CLIPS:
        clips.append({"start": round(t, 2), "end": round(t + chunk, 2), "score": 0, "text": ""})
        t += chunk
    return clips

def detect_face_center(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]
        return ((x + fw // 2) / w, (y + fh // 2) / h)
    return None

def extract_clip(video_path, start, end, output_path, clip_index):
    duration = end - start
    face = detect_face_center(video_path, start + duration / 2)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", video_path
    ], capture_output=True, text=True)
    info = json.loads(result.stdout)
    orig_w = info["streams"][0]["width"]
    orig_h = info["streams"][0]["height"]
    target_ratio = 9 / 16
    orig_ratio = orig_w / orig_h
    if orig_ratio > target_ratio:
        crop_h = orig_h
        crop_w = int(orig_h * target_ratio)
        crop_x = max(0, min(int(face[0] * orig_w) - crop_w // 2, orig_w - crop_w)) if face else (orig_w - crop_w) // 2
        crop_y = 0
    else:
        crop_w = orig_w
        crop_h = int(orig_w / target_ratio)
        crop_x = 0
        crop_y = max(0, min(int(face[1] * orig_h) - crop_h // 2, orig_h - crop_h)) if face else (orig_h - crop_h) // 2
    vf = (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
    )
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t", str(duration), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        output_path
    ], capture_output=True)

def burn_captions(clip_path, segments, clip_start, clip_end):
    srt_path = clip_path.replace(".mp4", ".srt")
    output_with_captions = clip_path.replace(".mp4", "_captioned.mp4")
    relevant = [s for s in segments if s["start"] < clip_end and s["end"] > clip_start]
    if not relevant:
        return
    def fmt(t):
        h, m, s, ms = int(t//3600), int((t%3600)//60), int(t%60), int((t%1)*1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(srt_path, "w") as f:
        for i, seg in enumerate(relevant, 1):
            s = max(0, seg["start"] - clip_start)
            e = min(clip_end - clip_start, seg["end"] - clip_start)
            f.write(f"{i}\n{fmt(s)} --> {fmt(e)}\n{seg.get('text','').strip()}\n\n")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=18,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2,Shadow=1,Alignment=2,MarginV=80'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "copy",
        output_with_captions
    ], capture_output=True)
    if result.returncode == 0 and os.path.exists(output_with_captions):
        os.replace(output_with_captions, clip_path)
    if os.path.exists(srt_path):
        os.remove(srt_path)

def process_video(input_path, output_folder):
    print(f"Processing: {input_path}")
    duration = get_video_duration(input_path)
    segments = transcribe_video(input_path)
    clips = find_best_clips(segments, duration)
    output_files = []
    for i, clip in enumerate(clips):
        clip_name = f"clip_{i+1:02d}.mp4"
        clip_path = os.path.join(output_folder, clip_name)
        extract_clip(input_path, clip["start"], clip["end"], clip_path, i)
        if os.path.exists(clip_path):
            burn_captions(clip_path, segments, clip["start"], clip["end"])
            output_files.append({
                "filename": clip_name,
                "start": clip["start"],
                "end": clip["end"],
                "duration": round(clip["end"] - clip["start"], 1)
            })
    print(f"Done! Generated {len(output_files)} clips.")
    return output_files
