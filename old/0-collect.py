# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.8.12 ('ai')
#     language: python
#     name: python3
# ---

# %%
import pyautogui
import cv2
import mouse
import keyboard
import threading

# %%
# Specify resolution
resolution = (1920, 1080)

# Specify video codec
codec = cv2.VideoWriter_fourcc(*"XVID")

# Specify name of Output file
filename = "Recording.avi"

# Specify frames rate. We can choose
# any value and experiment with it
fps = 60.0

# Creating a VideoWriter object
out = cv2.VideoWriter(filename, codec, fps, resolution)


def record_screen(rec_name: str):
    while True:

        # Take screenshot using PyAutoGUI
        img = pyautogui.screenshot()

        # Convert the screenshot to a numpy array
        frame = np.array(img)

        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write it to the output file
        out.write(frame)

        # Optional: Display the recording screen
        # cv2.imshow('Live', frame)

        # Stop recording when we press 'a'
        if cv2.waitKey(1) == ord('a'):
            break

    # Release the Video writer
    out.release()

    # Destroy all windows
    # cv2.destroyAllWindows()


# %%
mouse_events = []

mouse.hook(mouse_events.append)
keyboard.start_recording()  # Starting the recording

keyboard.wait("a")

mouse.unhook(mouse_events.append)
# Stopping the recording. Returns list of events
keyboard_events = keyboard.stop_recording()
# %%
mouse_events

# %%

s_thread = threading.Thread(target=record_screen, args=('human',))
k_thread = threading.Thread(target=lambda: keyboard.play(keyboard_events))
m_thread = threading.Thread(target=lambda: mouse.play(mouse_events))

k_thread.start()
m_thread.start()

# waiting for both threadings to be completed

k_thread.join()
m_thread.join()
