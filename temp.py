from tkinter import Tk, filedialog, Button, Label, Entry
import cv2
import numpy as np
import tensorflow as tf
import openpyxl
import time

interpreter = tf.lite.Interpreter(model_path="C:/Users/Third/Downloads/custom_model_lite2/custom_model_lite/detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Create a GUI window
root = Tk()
root.title("Video Processing")
root.geometry("300x150")
root.attributes('-topmost', True)  # Keep the window always on top

# Create a label for the finish time
finish_label = Label(root, text="Estimated Finish Time: --:--")
finish_label.pack(pady=5)

# Create a label and entry for frame skip interval
frame_skip_label = Label(root, text="Frame Skip Interval:")
frame_skip_label.pack(pady=5)
frame_skip_entry = Entry(root)
frame_skip_entry.pack(pady=5)
frame_skip_entry.insert(0, "1")  # Default value

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  
    return np.expand_dims(normalized_frame, axis=0)

def start_processing():
    # Use the value from the entry field
    frame_skip_interval = int(frame_skip_entry.get())

    # Use a file dialog to choose the video file
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))

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

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip_interval == 0:
            resized_frame = cv2.resize(frame, (small_width, small_height))
            
            input_data = preprocess_frame(resized_frame)
            
            input_data = np.float32(input_data)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            if output_data[0][0] > 0.2:
                confidence_percentage = int(output_data[0][0] * 100)
                text = f"Object: {confidence_percentage}%"
                cv2.rectangle(resized_frame, (50, 50), (150, 150), (0, 255, 0), 2)
                cv2.putText(resized_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if 20 <= output_data[0][0] * 100 <= 100:  
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
            
            out.write(resized_frame)
            
            cv2.imshow('Frame', resized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

        # Calculate and update the estimated finish time every second
        if frame_number % fps == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_frame = elapsed_time / frame_number
            frames_remaining = total_frames - frame_number
            estimated_time_remaining = avg_time_per_frame * frames_remaining
            eta_minutes = int(estimated_time_remaining // 60)
            eta_seconds = int(estimated_time_remaining % 60)
            eta_text = f"{eta_minutes:02}:{eta_seconds:02}"
            finish_label.config(text=f"Estimated Finish Time: {eta_text}")
            root.update()  # Force update of the GUI

    finish_time = time.time()
    finish_minutes = int((finish_time - start_time) // 60)
    finish_seconds = int((finish_time - start_time) % 60)
    finish_text = f"Estimated Finish Time: {finish_minutes:02}:{finish_seconds:02}"
    finish_label.config(text=finish_text)
    root.update()

    wb.save('detection_results.xlsx')

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Create start button
start_button = Button(root, text="Start Processing", command=start_processing)
start_button.pack(pady=5)

root.mainloop()
