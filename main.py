import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.models.model import SegmentationInference

segmenter = SegmentationInference()

def process_image(image, segmenter):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    result = segmenter.predict(image)
    return result.plot()

def process_video_with_progress(video_path, segmenter, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        processed_frame = process_image(frame, segmenter)
        out.write(processed_frame)

        progress_bar.progress((i + 1) / total_frames)

    cap.release()
    out.release()
    return output_path

st.title("Интерфейс сегментации YOLO")
st.write("Загрузите изображение или видео для обработки")

input_type = st.radio("Тип данных для обработки:", ("Изображение", "Видео"))

if input_type == "Изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Оригинальное изображение", use_column_width=True)
        
        processed_image = process_image(image, segmenter)
        st.image(processed_image, caption="Обработанное изображение", use_column_width=True)
        
        output_image_path = "processed_image.png"
        cv2.imwrite(output_image_path, cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        with open(output_image_path, "rb") as file:
            st.download_button(
                label="Скачать обработанное изображение",
                data=file,
                file_name="processed_image.png",
                mime="image/png"
            )

elif input_type == "Видео":
    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        st.video("temp_video.mp4")

        st.write("Обработка видео...")
        output_video_path = process_video_with_progress("temp_video.mp4", segmenter)
        
        with open(output_video_path, "rb") as file:
            st.download_button(
                label="Скачать обработанное видео",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )