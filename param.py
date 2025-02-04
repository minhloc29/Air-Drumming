import cv2
from my_utils import process_image
import simpleaudio as sa

upper_lip_index = 13  
lower_lip_index = 14 

snare_img = cv2.imread("toolkits/snare.jpg", cv2.IMREAD_UNCHANGED) 
hihat_img = cv2.imread("toolkits/hihat.jpg", cv2.IMREAD_UNCHANGED) 
crash_img = cv2.imread("toolkits/crash.jpg", cv2.IMREAD_UNCHANGED) 
cymbal_img = cv2.imread("toolkits/cymbal.jpg", cv2.IMREAD_UNCHANGED) 

snare_sound = sa.WaveObject.from_wave_file('sound/snare.wav')
hihat_sound = sa.WaveObject.from_wave_file('sound/hi_hat.wav')
cymbal_sound = sa.WaveObject.from_wave_file('sound/cymbal.wav')
crash_sound = sa.WaveObject.from_wave_file('sound/crash.wav')
kick_sound = sa.WaveObject.from_wave_file('sound/kick_sound.wav')

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

SIZE = (300, 300) #SIZE OF AN REGION
SOUNDS = [
    {  
        'name': 'hi-hat',
        'rect': (200, 200, 500, 500),  # x1, y1, x2, y2
        'sound': hihat_sound,
        'hand': 'Left',
        'state': False,
        'image': process_image(hihat_img, SIZE)
    },
    {  
        'name': 'cymbal',
        'rect': (w - 900, 600, w - 600, 900),  # x1, y1, x2, y2
        'sound': cymbal_sound,
        'hand': 'Left',
        'state': False,
        'image': process_image(cymbal_img, SIZE)
    },
    {  
        'name': 'crash',
        'rect': (600, 600, 900, 900),  # x1, y1, x2, y2
        'sound': crash_sound,
        'hand': 'Left',
        'state': False,
        'image': process_image(crash_img, SIZE)
    },
    {  
        'name': 'snare',
        'rect': (w - 500, 200, w - 200, 500),
        'sound': snare_sound,
        'hand': 'Right',
        'state': False,
        'image': process_image(snare_img, SIZE)
    },
    {  
        'name': 'kick',
        'sound': kick_sound,
        'state': False,
    }
]