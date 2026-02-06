import cv2
import numpy as np
import dlib
from math import hypot
import random
keyboard = np.zeros((600, 1000, 3), np.uint8)
font = cv2.FONT_HERSHEY_PLAIN
frames = 0


def draw_letters(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    width = 200
    height = 200
    th = 3 # thickness
    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

def draw_letters1(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 6:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 7:
        x = 200
        y = 200
    elif letter_index == 2:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200

    width = 200
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)


def draw_letters2(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 4:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 6:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 8:
        x = 200
        y = 200
    elif letter_index == 2:
        x = 400
        y = 200
    elif letter_index == 7:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200

    width = 200
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

def draw_letters3(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 8:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 9:
        x = 0
        y = 200
    elif letter_index == 7:
        x = 200
        y = 200
    elif letter_index == 2:
        x = 400
        y = 200
    elif letter_index == 6:
        x = 600
        y = 200
    elif letter_index == 5:
        x = 800
        y = 200

    width = 200
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)


def draw_letters4(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 6:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 7:
        x = 800
        y = 0
    elif letter_index == 4:
        x = 0
        y = 200
    elif letter_index == 8:
        x = 200
        y = 200
    elif letter_index == 5:
        x = 400
        y = 200
    elif letter_index == 2:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200

    width = 200
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)


def draw_letters5(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 3:
        x = 200
        y = 0
    elif letter_index == 6:
        x = 400
        y = 0
    elif letter_index == 1:
        x = 600
        y = 0
    elif letter_index == 2:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 7:
        x = 200
        y = 200
    elif letter_index == 4:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200

    width = 200
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)



def draw_menu():
    rows, cols, _ = keyboard.shape
    cv2.putText(keyboard, "PIN", (80, 300), font, 6, (255, 255, 255), 5)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
    
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks,frame,gray):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio
    
def process():
    global frames
    rnd=random.randrange(0, 5)
    cap = cv2.VideoCapture(0)
    board = np.zeros((300, 1400), np.uint8)
    board[:] = 255
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Keyboard settings    
    keys_set_1 = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4",5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    keys_set_2 = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4",5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    # Counters    
    letter_index = 0
    blinking_frames = 0
    frames_to_blink = 4
    frames_active_letter = 18
    # Text and keyboard settings
    text = ""
    keyboard_selected = "left"
    last_keyboard_selected = "left"
    select_keyboard_menu = True
    keyboard_selection_frames = 0
    while True:
        _, frame = cap.read()
        rows, cols, _ = frame.shape
        keyboard[:] = (26, 26, 26)
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Draw a white space for loading bar
        if select_keyboard_menu is True:
            draw_menu()
        keys_set = keys_set_1
        active_letter = keys_set[letter_index]
        faces = detector(gray)  # Face detection
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye, right_eye = eyes_contour_points(landmarks)
            # Detect blinking
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            # Eyes color
            cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
            if select_keyboard_menu is True:
                # Detecting gaze to select Left or Right keybaord
                gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks,frame,gray)
                gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks,frame,gray)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
                if gaze_ratio <= 0.9:
                    keyboard_selected = "right"
                    keyboard_selection_frames += 1
                    if keyboard_selection_frames == 15:  # If Kept gaze on one side more than 15 frames, move to keyboard
                        select_keyboard_menu = False
                        frames = 0   # Set frames count to 0 when keyboard selected
                        keyboard_selection_frames = 0
                    if keyboard_selected != last_keyboard_selected:
                        last_keyboard_selected = keyboard_selected
                        keyboard_selection_frames = 0
                else:
                    keyboard_selected = "left"
                    keyboard_selection_frames += 1
                    if keyboard_selection_frames == 15:  # If Kept gaze on one side more than 15 frames, move to keyboard
                        select_keyboard_menu = False
                        frames = 0  # Set frames count to 0 when keyboard selected
                    if keyboard_selected != last_keyboard_selected:
                        last_keyboard_selected = keyboard_selected
                        keyboard_selection_frames = 0

            else:
                if blinking_ratio > 5:   # Detect the blinking to select the key that is lighting up
                    blinking_frames += 1
                    frames -= 1
                    # Show green eyes when closed
                    cv2.polylines(frame, [left_eye], True, (255, 255, 0), 2)
                    cv2.polylines(frame, [right_eye], True, (255, 255, 0), 2)

                    # Typing letter
                    if blinking_frames == frames_to_blink:
                        if active_letter != "<" and active_letter != "_":
                            text += active_letter
                        if active_letter == "_":
                            text += " "
                        select_keyboard_menu = True

                else:
                    blinking_frames = 0
        # Display letters on the keyboard
        if select_keyboard_menu is False:
            if frames == frames_active_letter:
                letter_index += 1
                frames = 0
            if letter_index == 10:
                letter_index = 0
            for i in range(10):
                if i == letter_index:
                    light = True
                else:
                    light = False
                if rnd==0:            
                    draw_letters(i, keys_set[i], light)
                if rnd==1:            
                    draw_letters1(i, keys_set[i], light)
                if rnd==2:            
                    draw_letters2(i, keys_set[i], light)
                if rnd==3:            
                    draw_letters3(i, keys_set[i], light)
                if rnd==4:            
                    draw_letters4(i, keys_set[i], light)
                if rnd==5:            
                    draw_letters5(i, keys_set[i], light)
        #print("Text value",text)  # Show the text we're writing on the board
        percentage_blinking = blinking_frames / frames_to_blink  # Blinking loading bar
        loading_x = int(cols * percentage_blinking)
        cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
        cv2.imshow("Frame", frame)
        cv2.imshow("Virtual keyboard", keyboard)
        key = cv2.waitKey(1)
        if text !="":
            break    
    cap.release()
    cv2.destroyAllWindows()
    return text