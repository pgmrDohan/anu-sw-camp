from typing import List
import os
import random
import datetime
import secrets

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model
import trafilatura
from bs4 import BeautifulSoup

app = FastAPI()

# 이미지 및 템플릿 디렉토리 설정
IMG_DIR = './photo/'
templates = Jinja2Templates(directory="templates")

# 얼굴 인식을 위한 Haar Cascade 로드
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# 표정 레이블과 모델 로드
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model('./emotion_model.hdf5')

# 이미지의 정적 디렉토리 마운트
app.mount("/images", StaticFiles(directory=IMG_DIR), name="photo")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    # 기본 페이지 렌더링
    return templates.TemplateResponse(name="index.html", context={"request": request})

@app.post('/upload-images')
async def upload_images(in_files: List[UploadFile] = File(...)):
    # 업로드된 파일 처리
    for file in in_files:
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        saved_file_name = f"{current_time}_{secrets.token_hex(8)}"
        file_location = os.path.join(IMG_DIR, saved_file_name)

        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        # 저장된 이미지 경로로 AI 처리로 리다이렉션
        return RedirectResponse(url=f"/ai?img={file_location}", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/ai")
async def analyze_image(request: Request, img: str):
    # 업로드된 이미지 읽기 및 처리
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지에서 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 감지된 각 얼굴에 대해 감정 분석
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64)) / 255.0  # 크기 조정 및 정규화

        face_roi = np.expand_dims(face_roi, axis=(0, -1))  # 배치 및 채널 차원 추가

        # 감정 예측
        output = model.predict(face_roi)[0]
        expression_label = expression_labels[np.argmax(output)]

    # 감지된 감정에 따라 음악 추천 가져오기
    res = trafilatura.fetch_url(f'https://8tracks.com/explore/{expression_label.lower()}/hot')
    soup = BeautifulSoup(res, "html.parser")
    titles = soup.find_all('div', attrs={'class': 'mix_square'})

    # 음악 믹스 링크 수집
    links = [title.find('a', class_='mix_url')['href'] for title in titles]

    # 랜덤으로 선택된 음악 믹스 URL로 리다이렉션
    music_mix_url = f"https://8tracks.com{random.choice(links)}"

    # 이미지 파일 삭제
    os.remove(img)

    return RedirectResponse(url=music_mix_url)
