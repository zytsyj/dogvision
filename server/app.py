import os
import cv2
import uvicorn
import base64
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List

app = FastAPI()
DATABASE_URL = "mysql+pymysql://root:zyt622312@127.0.0.1:3306/summars?charset=utf8mb4"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Worker(Base):
    __tablename__ = "workers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    num_violation = Column(Integer)
    pho_violation = Column(LargeBinary)  # 存储二进制数据


Base.metadata.create_all(bind=engine)


class WorkerCreate(BaseModel):
    name: str
    num_violation: int


class WorkerResponse(BaseModel):
    id: int
    name: str
    num_violation: int
    pho_violation: str

    class Config:
        orm_mode = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/workers/", response_model=WorkerResponse)
def create_worker(name: str, num_violation: int, pho_violation: UploadFile = File(...), db: Session = Depends(get_db)):
    file_data = pho_violation.file.read()
    db_worker = Worker(name=name, num_violation=num_violation, pho_violation=file_data)
    db.add(db_worker)
    db.commit()
    db.refresh(db_worker)
    return db_worker


@app.get("/workers/", response_model=List[WorkerResponse])
def read_workers(db: Session = Depends(get_db)):
    workers = db.query(Worker).all()
    for worker in workers:
        worker.pho_violation = f"data:image/jpeg;base64,{base64.b64encode(worker.pho_violation).decode('utf-8')}"
    return workers


@app.get("/workers/{worker_id}", response_model=WorkerResponse)
def read_worker(worker_id: int, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if worker is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker.pho_violation = f"data:image/jpeg;base64,{base64.b64encode(worker.pho_violation).decode('utf-8')}"
    return worker


@app.put("/workers/{worker_id}", response_model=WorkerResponse)
def update_worker(worker_id: int, name: str, num_violation: int, pho_violation: UploadFile = File(None),
                  db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if worker is None:
        raise HTTPException(status_code=404, detail="Worker not found")

    worker.name = name
    worker.num_violation = num_violation

    if pho_violation:
        file_data = pho_violation.file.read()
        worker.pho_violation = file_data

    db.commit()
    db.refresh(worker)
    worker.pho_violation = f"data:image/jpeg;base64,{base64.b64encode(worker.pho_violation).decode('utf-8')}"
    return worker


@app.delete("/workers/{worker_id}", response_model=WorkerResponse)
def delete_worker(worker_id: int, db: Session = Depends(get_db)):
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if worker is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    db.delete(worker)
    db.commit()
    return worker


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    camera = cv2.VideoCapture(0)  # 打开摄像头
    while True:
        success, frame = camera.read()  # 读取摄像头帧
        frame = cv2.flip(frame, 1)
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)  # 将帧转化为JPEG格式
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 生成帧数据


@app.get("/frame_base64")
def get_frame_base64():
    camera = cv2.VideoCapture(0)  # 打开摄像头
    success, frame = camera.read()  # 读取摄像头帧
    if not success:
        raise HTTPException(status_code=500, detail="Failed to capture image")
    frame = cv2.flip(frame, 1)
    ret, buffer = cv2.imencode('.jpg', frame)  # 将帧转化为JPEG格式
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    base64_data = base64.b64encode(buffer).decode('utf-8')
    html_str = f"data:image/jpg;base64,{base64_data}"
    return JSONResponse(content={"image": html_str})

@app.get('/yolo')
def video_feed():
    return StreamingResponse(run(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=6574)
