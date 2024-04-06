from tkinter import Tk, filedialog, Button, Label, Entry
from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import time

# Variable for whether or not the video is playback is paused. True for playing, False for Paused
playing = True

""""frame_divisor = 2

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
"""
def open_file(): 
    open_file = Tk()
    open_file.withdraw()  # Hide the main window
    input_video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    open_file.destroy()  # Destroy the root window after selection"""
def set_model():
    # Write function later. Function should open up file dialog and set model path, see open_file function
    pass
def set_frame_skip_interval():
    # Write function later. Function should open up window to set frame skip interval
    pass
def set_output_destination():
    # Write function later. Function should open up window to set output directory for video & spreadsheet
    pass
def toggle_playback(playback_button):
    global playing
    if playing:
        playback_button.config(image=pause_image)
        playing = False
    else:
        playback_button.config(image=play_image)
        playing = True

# Set up root window
root = Tk()
root.geometry("1000x800+500+100")
root.title("Roc")

# Create images to be used later.
play_image = Image.open("sprint-four/codebase/assets/play.png")
play_image = play_image.resize((25, 25))
play_image = ImageTk.PhotoImage(play_image)
pause_image = Image.open("sprint-four/codebase/assets/pause.png")
pause_image = pause_image.resize((25, 25))
pause_image = ImageTk.PhotoImage(pause_image)

# Set up left frame for video playback.
left_frame = ttk.Frame(root, padding="3 3 12 12", width=500, height=800)
left_frame.pack(side="left", anchor=NW, padx=25, pady=25)

# Set up right frame for data display
right_frame = ttk.Frame(root, padding="3 3 12 12", width=500, height=800)
right_frame.pack(side="left", anchor=NW, padx=25, pady=25)

# Create Menu
menu = tk.Menu(root)

# Create File Menu
file_menu = tk.Menu(menu, tearoff=False)
file_menu.add_command(label="Open File", command=open_file) # Logic partially implemented. 
recent_menu = tk.Menu(file_menu, tearoff=False)
file_menu.add_cascade(label="Open Recent", menu=recent_menu) # (Logic for this command is not implemented yet.)
recent_menu.add_command(label="Example Recent File") # placeholder for visual test
menu.add_cascade(label="File", menu=file_menu)

# Create Settings Menu
settings_menu = tk.Menu(menu, tearoff=False)
settings_menu.add_command(label="Select Model", command=set_model) # (Logic for this command is not implemented yet.)
settings_menu.add_separator()
settings_menu.add_command(label="Set Frame Skip Interval", command=set_frame_skip_interval) # (Logic for this command is not implemented yet.)
settings_menu.add_command(label="Set Output Destination", command=set_output_destination)
settings_menu.add_separator()
settings_menu.add_checkbutton(label="Frame Skip") # Logic not implemented
settings_menu.add_checkbutton(label="Output Video File") # Logic not implemented
settings_menu.add_checkbutton(label="Output Timestamp Spreadsheet") # Logic not implemented
menu.add_cascade(label="Settings", menu=settings_menu)

# Add menu to root window
root.config(menu=menu)

# Set up Layout on left side
# Placeholder image to represent video frame
ex_img = Image.open("sprint-four/codebase/assets/video_example.png")
ex_img = ex_img.resize((480, 270))
resized_ex_img = ImageTk.PhotoImage(ex_img)
image_widget = ttk.Label(left_frame, image=resized_ex_img)
image_widget.pack(side=TOP, anchor=N)
# playback button
playback_button = ttk.Button(left_frame, image=pause_image, command=lambda: toggle_playback(playback_button),) # logic not implemented
playback_button.pack(side=TOP, anchor=W, padx=200)

# Set up Layout on right side

# variables for parts of labels that need to update, set to placeholder values for now, fix later
timestamp_value = "1:23:00"
current_frame_value = 117
confidence_value = .75

#Top Section
timestamp_label = ttk.Label(right_frame, text=f"Timestamp: {timestamp_value}", font=("Terminal", 20)) # Logic for updating now implemented yet
timestamp_label.pack(side=TOP, anchor=NW)
current_frame_label = ttk.Label(right_frame, text=f"Current Frame: {current_frame_value}", font=("Terminal", 20)) # Logic for updating now implemented yet
current_frame_label.pack(side=TOP, anchor=NW)
confidence_label = ttk.Label(right_frame, text=f"Confidence Level: {confidence_value:.0%}", font=("Terminal", 20)) # Logic for updating now implemented yet
confidence_label.pack(side=TOP, anchor=NW)

#Bottom Section (Split into separate frames?) Note that the top of this section of info is 200 below the top of the right frame
arrival_departure_label = ttk.Label(right_frame, text="Arrivals & Departures", font=("Terminal", 20))
arrival_departure_label.pack(side=TOP, anchor=NW, pady=200)
arrival_departure_separator = ttk.Separator(right_frame, orient="horizontal")
arrival_departure_separator.pack(side=TOP)


root.mainloop()