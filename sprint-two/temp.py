import cv2
import numpy as np
import tensorflow as tf
import openpyxl
import json
import time

# start time tracking (for tracking performance)
start = time.time()

# load configuration settings
f = open('sprint-two/config.json')
settings = json.load(f)
input_model_path = settings["input_model_path"]
input_video_path = settings["input_video_path"]
output_video_path = settings["output_video_path"]
output_timestamps_path = settings["output_timestamps_path"]
frame_divisor = int(settings["frame_divisor"]) # Only frames with a frame number divisible by this number will be processed (1 for all frames, this is for optimization) 
confidence_threshold = float(settings["confidence_threshold"])
f.close()

print(confidence_threshold)


# create interpreter and load with pre-trained model 
interpreter = tf.lite.Interpreter(model_path=input_model_path)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details() # list of dictionaries, each dictionary has details about an input tensor
output_details = interpreter.get_output_details() # list of dictionaries, each dictionary has details about an input tensor
input_shape = input_details[0]['shape'] # array of shape of input tensor

def preprocess_frame(frame):    # function for processing frames from capture
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  
    return np.expand_dims(normalized_frame, axis=0)

def check_frame(processed_frame):   # function for sending frame to interpreter to be checked for bird
        processed_frame = np.float32(processed_frame)
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

def thresholding(checked_frame, confidence_start):    # function for thresholding based on confidence of model
        if checked_frame[0][0] > confidence_threshold:  # only show box if confidence is over 50%
            confidence_percentage = int(checked_frame[0][0] * 100)
            # stamp confidence percentage onto up left-hand corner of frame
            text = f"Object: {confidence_percentage}%"
            cv2.rectangle(checked_frame, (50, 50), (150, 150), (0, 255, 0), 2)
            cv2.putText(checked_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if  confidence_threshold <= checked_frame[0][0]: # if we're above 50% confidence, start a timestamp if one hasn't been started
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
        return checked_frame, confidence_start


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
    # only check every fifth frame to increase speed
    
    # process frame with preprocess_frame
    processed_frame = preprocess_frame(frame)

    if frame_number % frame_divisor == 0:
        # send processed frame to interpreter be checked for bird
        checked_frame = check_frame(processed_frame)
        # send checked frame to thresholding to see if confidence is high enough
        # if so, handle confidence stamp and timestamp
        final_frame, confidence_start = thresholding(checked_frame, confidence_start)
        # write the altered frame to the output video file.
        out.write(final_frame)

    # show altered frames as video on screen 
    # cv2.imshow('Frame', frame)

    # if q key is pressed, end early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

# save completed workbook to output excel file
wb.save(output_timestamps_path)

# close everything
cap.release()
out.release()
cv2.destroyAllWindows()

# get execution time for entire program (for tracking performance)
end = time.time()
print(f"Program took: {(end-start)} seconds.")
print(f"Program took: {(end-start)*1000} milliseconds.")
