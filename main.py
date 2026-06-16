import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from clipper import process_video
import asyncio

app = FastAPI(title="Clipify - AI Video Clipper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
STATIC_DIR = "static"

for d in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

jobs = {}

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/mpeg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only video files are accepted (MP4, MOV, AVI, WebM)")

    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_folder, exist_ok=True)

    jobs[job_id] = {"status": "processing", "clips": [], "error": None}
    background_tasks.add_task(run_processing, job_id, input_path, output_folder)

    return JSONResponse({"job_id": job_id})

async def run_processing(job_id: str, input_path: str, output_folder: str):
    try:
        loop = asyncio.get_event_loop()
        clips = await loop.run_in_executor(None, process_video, input_path, output_folder)
        jobs[job_id]["clips"] = clips
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(jobs[job_id])

@app.get("/download/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)
