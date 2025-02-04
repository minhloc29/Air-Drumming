import cv2
import math
from threading import Thread

def process_image(img, target_size):
    # If image has alpha channel, use it directly
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        # Create alpha channel (white background -> transparent)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Resize image to target drum zone size
    resized = cv2.resize(img, (target_size[0], target_size[1]))
    alpha = cv2.resize(alpha, (target_size[0], target_size[1]))
    
    # Convert to BGRA if not already
    if resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
    
    resized[:, :, 3] = alpha  # Apply alpha channel
    return resized

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def play_sound_async(sound):
    def play():
        play_obj = sound.play()
        play_obj.wait_done()
    Thread(target=play).start()