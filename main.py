from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading
from video_engine import VideoEngine

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize video engine
video_engine = VideoEngine()
thread = None

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/start")
async def start_streaming():
    global thread
    if video_engine.streaming:
        raise HTTPException(status_code=400, detail="Streaming already active")
    thread = threading.Thread(target=video_engine.run)
    thread.start()
    return {"status": "Streaming started"}

@app.post("/stop")
async def stop_streaming():
    if not video_engine.streaming:
        raise HTTPException(status_code=400, detail="Streaming not active")
    video_engine.stop_streaming()
    global thread
    thread.join()
    return {"status": "Streaming stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)