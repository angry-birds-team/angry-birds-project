import cv2
import numpy as np
import tensorflow as tf
import openpyxl
import time
from tkinter import Tk, filedialog

interpreter = tf.lite.Interpreter(model_path="PATH TO MODEL")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  
    return np.expand_dims(normalized_frame, axis=0)

# Use a file dialog to choose the video file
root = Tk()
root.withdraw()  # Hide the main window
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
root.destroy()  # Destroy the root window after selection

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Specify the desired smaller width and height
small_width = width // 2
small_height = height // 2

out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (small_width, small_height))

wb = openpyxl.Workbook()
sheet = wb.active
sheet.append(["Start Time (min:sec)", "End Time (min:sec)"])

frame_number = 0
confidence_start = None

# Define the frame skip interval
frame_skip_interval = 2  # Skip every 2 frames

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_skip_interval == 0:
        # Resize the frame to the smaller size
        resized_frame = cv2.resize(frame, (small_width, small_height))
        
        input_data = preprocess_frame(resized_frame)
        
        input_data = np.float32(input_data)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        if output_data[0][0] > 0.5:
            confidence_percentage = int(output_data[0][0] * 100)
            text = f"Object: {confidence_percentage}%"
            cv2.rectangle(resized_frame, (50, 50), (150, 150), (0, 255, 0), 2)
            cv2.putText(resized_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if 50 <= output_data[0][0] * 100 <= 100:  
            if confidence_start is None:
                confidence_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else:
            if confidence_start is not None:
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                minutes_start = int(confidence_start // 60)
                seconds_start = int(confidence_start % 60)
                minutes_end = int(timestamp_sec // 60)
                seconds_end = int(timestamp_sec % 60)
                timestamp_start = f"{minutes_start:02}:{seconds_start:02}"
                timestamp_end = f"{minutes_end:02}:{seconds_end:02}"
                sheet.append([timestamp_start, timestamp_end])
                confidence_start = None
        
        elapsed_time = time.time() - start_time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        elapsed_text = f"Elapsed Time: {elapsed_minutes:02}:{elapsed_seconds:02}"
        cv2.putText(resized_frame, elapsed_text, (10, small_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(resized_frame)
        
        cv2.imshow('Frame', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_number += 1

wb.save('detection_results.xlsx')

cap.release()
out.release()
cv2.destroyAllWindows()