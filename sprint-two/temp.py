import cv2
import numpy as np
import tensorflow as tf
import openpyxl

# create interpreter and load with pre-trained model 
interpreter = tf.lite.Interpreter(model_path="sprint-two/custom_model_lite/detect.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details() # list of dictionaries, each dictionary has details about an input tensor
output_details = interpreter.get_output_details() # list of dictionaries, each dictionary has details about an input tensor
input_shape = input_details[0]['shape'] # array of shape of input tensor

def preprocess_frame(frame):    # function for processing frames from capture
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 255.0  
    return np.expand_dims(normalized_frame, axis=0)

# start capture from video file
cap = cv2.VideoCapture('sprint-two/test_video.mp4')

# get fps, dimensions, total frames from video capture
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# output capture to video file
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

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
    
    # process frame with preprocess_frame
    input_data = preprocess_frame(frame)
    
    # send processed frame to interpreter be checked for bird
    input_data = np.float32(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_data[0][0] > 0.5:  # only show box if confidence is over 50%
        confidence_percentage = int(output_data[0][0] * 100)
        # stamp confidence percentage onto up left-hand corner of fram
        text = f"Object: {confidence_percentage}%"
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if 50 <= output_data[0][0] * 100 <= 100: # if we're above 50% confidence, start a timestamp if one hasn't been started
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
    # write the altered frame to the output video file.
    out.write(frame)

    # show altered frames as video on screen 
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

# save completed workbook to output excel file
wb.save('detection_results.xlsx')

# close everything
cap.release()
out.release()
cv2.destroyAllWindows()