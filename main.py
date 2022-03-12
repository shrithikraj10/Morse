from cv2 import cv2
import numpy as np
import dlib
from math import hypot
import time
import pyttsx3

blinks = []
sentence = ''

BLINK_FRAME_COUNT = 3

blink_frame = 0

EYE_OPEN_RATIO = 3
EYE_OPEN_TIME_DIFFERENCE = 2
AVERAGE_BLINKING_RATIO = 4.9
TIME_FOR_LONG_BLINK = 0.3
TIME_FOR_SHORT_BLINK = 0.2

# Morse Codes
morse_codes = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "----": "Switch Mode"
}

# Predefined Words
predefined_words = {
    ".": "Yes",
    "-": "No",
    ".-": "Blank",                              # Blank
    "-.": "Blank",                              # Blank
    "--": "Medicine",
    "..": "Water",
    "----": "Switch Mode"
}

dictionary_to_use = predefined_words

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN
sentence_window = cv2.namedWindow("Sentences")


# Returns the midpoint of the center top and center bottom inorder to make the vertical axis
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# Returns the ratio of the horizontal and vertical euclidean distance
def get_blinking_ratio(eye_points, facial_landmarks, frames):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    cv2.line(frames, left_point, right_point, (0, 255, 0), 2)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))

    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    cv2.line(frames, center_top, center_bottom, (0, 255, 0), 2)

    ver_line_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])
    hor_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])

    ratio = hor_line_length / ver_line_length
    return ratio


def words_window(outputs):
    cv2.putText(sentence_window, "Hello", (20, 20), font, 7, (255, 255, 255), 2)


# Checks if the code in the blinks array is present in the words dictionary and displays the words
def check_words():
    global sentence, dictionary_to_use
    letter = ''
    for char in blinks:
        if char == '':
            print(letter)
            if letter == '----':
                if dictionary_to_use == predefined_words:
                    print("Switching to Morse")
                    dictionary_to_use = morse_codes
                    sentence = ''
                else:
                    print("Switching to predefined words")
                    dictionary_to_use = predefined_words
                    engine = pyttsx3.init()
                    engine.say("Switching to predefined words")
                    engine.runAndWait()
                    sentence = ''
            else:
                if letter in dictionary_to_use:
                    sentence = sentence + dictionary_to_use[letter]
                    print(dictionary_to_use[letter])
                else:
                    print("No word exists for this code")
            letter = ''
        if char != '':
            letter += char
    print(sentence)
    words_window(sentence)

    # Here text is converted to speech, the code may not work for the first time since the welcome1.mp3 file has to be
    # created
    engine = pyttsx3.init()
    engine.say(sentence)
    engine.runAndWait()
    blinks.clear()


def calculate_blinks(start, is_first, eye_open_time, is_first_open):
    global blink_frame
    while True:
        did_grab_frame, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Gaze detection
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y),
                                        ], np.int32)

            right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                         (landmarks.part(43).x, landmarks.part(43).y),
                                         (landmarks.part(44).x, landmarks.part(44).y),
                                         (landmarks.part(45).x, landmarks.part(45).y),
                                         (landmarks.part(46).x, landmarks.part(46).y),
                                         (landmarks.part(47).x, landmarks.part(47).y),
                                         ], np.int32)

            # Separated Eyes
            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)

            cv2.polylines(mask, [left_eye_region, right_eye_region], True, 255, 2)
            cv2.fillPoly(mask, [left_eye_region, right_eye_region], 255)
            eyes = cv2.bitwise_and(gray, gray, mask=mask)
            bw_left_blinking_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks, eyes)
            bw_right_blinking_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks, eyes)

            avg_bw_br = (bw_left_blinking_ratio + bw_right_blinking_ratio) / 2

            # Checks if the eye is open. If open it notes the time it was open
            if avg_bw_br < EYE_OPEN_RATIO:
                if is_first_open:
                    eye_open_time = time.time()
                    print("Eye open detected")
                    calculate_blinks(start, is_first, eye_open_time, False)

            # Checks how long the eye is open. If open for long it puts a space in array
            if eye_open_time > 0:
                end = time.time()
                if end - eye_open_time > EYE_OPEN_TIME_DIFFERENCE:
                    print("Space appended")
                    blinks.append("")
                    cv2.putText(eyes, "Space", (50, 150), font, 7, (255, 0, 0), 2)
                    check_words()
                    calculate_blinks(start, is_first, 0, False)

            # Checks if a blink has occurred and notes the start time of the blink
            if avg_bw_br > AVERAGE_BLINKING_RATIO:
                if is_first:
                    start = time.time()
                if did_grab_frame:
                    blink_frame += 1                                                    # Count Number of Frames
                    print(f"Blink Frames: {blink_frame}")
                cv2.putText(eyes, "Blinking", (50, 150), font, 7, (255, 255, 255), 2)
                print(f"Avg Blink Ratio {avg_bw_br}")
                calculate_blinks(start, False, 0, False)

            # Checks the time when eye is opened after blink and checks difference between the closing and opening time
            # of the eye to segment blinking into short or long blink
            if avg_bw_br < AVERAGE_BLINKING_RATIO:
                end = time.time()
                timer = end - start

                if start > 0 and timer < EYE_OPEN_TIME_DIFFERENCE:
                    print(f"Timer {timer}")
                    if timer > TIME_FOR_LONG_BLINK and blink_frame >= BLINK_FRAME_COUNT:
                        print(f"Timer {timer} Long Blink")
                        cv2.putText(eyes, "Long Blink", (50, 200), font, 7, (255, 255, 255), 2)
                        blinks.append("-")
                        print("Setting blink_frame to 0")
                        blink_frame = 0
                        calculate_blinks(0, True, 0, True)

                    elif timer < TIME_FOR_SHORT_BLINK and blink_frame < BLINK_FRAME_COUNT:
                        print(f"Timer {timer} Short Blink")
                        cv2.putText(eyes, "Short Blink", (50, 250), font, 7, (255, 255, 255), 2)
                        blinks.append(".")
                        print("Setting blink_frame to 0")
                        blink_frame = 0
                        calculate_blinks(0, True, 0, True)
                    else:
                        print("Setting blink_frame to 0")
                        blink_frame = 0
                        calculate_blinks(0, True, 0, True)

                key = cv2.waitKey(10)
                if key == 27:
                    exit(0)
            cv2.imshow("Eyes", eyes)


calculate_blinks(0, True, 0, True)
cap.release()
cv2.destroyAllWindows()
