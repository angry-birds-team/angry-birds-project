import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = "2147483647"
print(os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'])
import cv2
import numpy as np
import tensorflow as tf
import openpyxl
import json
import time
from tkinter import Tk, filedialog, Button, Label, Entry

# start time tracking (for tracking performance)
start_time = time.time()

# load configuration settings
f = open('sprint-three/codebase/config.json')
settings = json.load(f)
input_model_path = settings["input_model_path"]
input_video_path = settings["input_video_path"]
output_video_path = settings["output_video_path"]
output_timestamps_path = settings["output_timestamps_path"]
frame_divisor = int(settings["frame_divisor"]) # Only frames with a frame number divisible by this number will be processed (1 for all frames, this is for optimization) 
confidence_threshold = float(settings["confidence_threshold"]) # confidence threshold should be between 0 and 1
f.close()

# create interpreter and load with pre-trained model 
interpreter = tf.lite.Interpreter(model_path=input_model_path)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details() # list of dictionaries, each dictionary has details about an input tensor
output_details = interpreter.get_output_details() # list of dictionaries, each dictionary has details about an input tensor
input_shape = input_details[0]['shape'] # array of shape of input tensor

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
frame_skip_entry.insert(0, str(frame_divisor))  # Default value

def preprocess_frame(frame):    # function for processing frames from capture
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  
    return np.expand_dims(normalized_frame, axis=0)

def check_frame(processed):   # function for sending frame to interpreter to be checked for bird
        processed = np.float32(processed)
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

def thresholding(checked_frame, confidence_start, frame):    # function for thresholding based on confidence of model
        if checked_frame[0][0] > confidence_threshold:  # only show box if confidence is over 50%
            confidence_percentage = int(checked_frame[0][0] * 100)
            # stamp confidence percentage onto up left-hand corner of frame
            text = f"Object: {confidence_percentage}%"
            cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if confidence_threshold <= checked_frame[0][0]: # if we're above confidence threshold, start a timestamp if one hasn't been started
            if confidence_start is None:
                confidence_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else: # if we're below 50% confidence, if we're in a timestamp, go ahead and add the timestamp to the worksheet and end it 
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
        return frame, confidence_start
root = Tk()
root.withdraw()  # Hide the main window
input_video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
root.destroy()  # Destroy the root window after selection

if __name__ == '__main__':
    # start capture from video file
    cap = cv2.VideoCapture(input_video_path)

    # get fps, dimensions, total frames from video capture
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    # output capture to video file
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # open workbook for collecting timestamps to later output to excel file
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.append(["Start Time (min:sec)", "End Time (min:sec)"])

    # initialize and set frame count and confidence to initial values
    frame_number = 0
    confidence_start = None

    while cap.isOpened():   # everything in this loop is being done as long as the capture is open.
        # get the return value and frame image data from cap
        ret, frame = cap.read()
        # if ret is false, no frame was grabbed, break
        if not ret:
            break
        
        # resize the frame
        # process frame with preprocess_frame
        # do not try and write this variable to a file, it's not compatible
        processed = preprocess_frame(frame)

        if frame_number % frame_divisor == 0: # only check every frame divisible by preset number to save time
            # send processed frame to interpreter be checked for bird
            checked = check_frame(processed)
            # send checked frame to thresholding to see if confidence is high enough
            # if so, handle confidence stamp and timestamp
            final_frame, confidence_start = thresholding(checked, confidence_start, frame)
            # write the altered frame to the output video file.
            out.write(final_frame)

        # show altered frames as video on screen 
        cv2.imshow('Frame', frame)

        # if q key is pressed, end early
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

    # save completed workbook to output excel file
    wb.save(output_timestamps_path)

    # close everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # get execution time for entire program (for tracking performance)
    end_time = time.time()
    print(f"Program took: {(end_time-start_time)} seconds.")
    print(f"Program took: {(end_time-start_time)*1000} milliseconds.")
