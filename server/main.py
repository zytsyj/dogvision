import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from vision.detect import run
app = FastAPI()

@app.get('/yolo')
def video_feed():
    return StreamingResponse(run(), media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)