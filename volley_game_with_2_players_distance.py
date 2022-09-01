import cv2
import random
import mediapipe as mp
from PIL import Image
import numpy as np
import time
import socket
import struct
import pickle
from pygame import mixer

color_red = (0,0,255) 
color_black = (0,0,0)
color_white = (255,255,255)
color_yellow = (0,180,255)
color_green = (0,255,0)

radius = 30
org_1, org_2  = (120,30), (440,30)
font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font_bold = cv2.FONT_HERSHEY_TRIPLEX

data = b""
data_2 = b""



def initialize_hand_tracking():
    mp_drawing = mp.solutions.drawing_utils          
    mp_drawing_styles = mp.solutions.drawing_styles  
    mp_hands = mp.solutions.hands 

    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    )
    return mp_drawing, mp_drawing_styles,mp_hands, hands

def find_quadrant(x,y):
    global quadrant
    if x<w and x>w/2 and y<h and y>h/2:
        quadrant="first"   
    elif x<w/2 and x>0 and y<h and y>h/2:
        quadrant="second"
    elif x<w/2 and x>0 and y<h/2 and y>0:
        quadrant="third"
    elif x<w and x>w/2 and y<h/2 and y>0:
        quadrant="fourth"    
    elif x==w/2 and x<w and y<h/2 and y>0:
        quadrant="fourth"    
    elif x>w/2 and x<w and y==h/2 and y<h:
        quadrant="first"   
    elif x==w/2 and x>0 and y>h/2 and y<h:
        quadrant="second"
    elif x<w/2 and x>0 and y==h/2 and y>0:
        quadrant="third"
    return quadrant

def ball_position():
    if quadrant=="first":
        rx_circle=random.randint(0+radius,w/2)
        ry_circle=random.randint(0+radius,h-radius)
    elif quadrant=="second":
        rx_circle=random.randint(w/2+radius,w-radius)
        ry_circle=random.randint(0+radius,h-radius)
    elif quadrant=="third":
        rx_circle=random.randint(w/2+radius,w-radius)
        ry_circle=random.randint(0+radius,h-radius)
    elif quadrant=="fourth":   
        rx_circle=random.randint(0+radius,w/2)
        ry_circle=random.randint(0+radius,h-radius)
    return (rx_circle,ry_circle)

def hand_tracking(actual_frame, hands, hit, score_1, score_2, run_circle):
    global bonus
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2RGB)   
    results = hands.process(actual_frame)
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [4,8,12,16,20]
            for i in landmarks:
                if not run_circle:
                    x = hand_landmarks.landmark[i].x * w   
                    y = hand_landmarks.landmark[i].y * h 
                    if x>(rx_circle-radius) and x<(rx_circle+radius) and y>(ry_circle-radius) and y<(ry_circle+radius):
                        run_circle = True
                        hit += 1
                        if i==8:
                            point = 1
                        elif i==12:
                            point = 2
                        elif i==4:
                            point = 3
                        elif i==20:
                            point = 4
                        elif i==16:
                            point = 5

                        if quadrant == 'first' or quadrant == 'fourth':
                            score_1 += point
                        elif quadrant == 'second' or quadrant=='third':
                            score_2 += point

                        if bonus:
                            if score_1 >= 25:
                                score_1 += 10
                                bonus = False
                            elif score_2 >= 25:
                                score_2 += 10
                                bonus = False

                mp_drawing.draw_landmarks(
                        actual_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                )
    return actual_frame, run_circle, hands,  score_1, score_2, hit, bonus

def draw_vertical_line(p1_y,p2_y,color,img):
    cv2.line(img, pt1=(int(w/2),p1_y), pt2=(int(w/2),p2_y), color=color, thickness=3)

def initialize_net(img):
    p1 = 0
    p2 = int(w/20)
    for i in range (0, 15):
        if i%2 == 0:
            color = color_black
        else:
            color = color_white
        draw_vertical_line(p1,p2,color,img)
        p1 += int(w/20)
        p2 += int(w/20)

def initialize_border(img):
    cv2.line(img, pt1=(0,0), pt2=(0,w), color=color_black, thickness=3)
    cv2.line(img, pt1=(0,0), pt2=(w,0), color=color_black, thickness=3)
    cv2.line(img, pt1=(w,0), pt2=(w,w), color=color_black, thickness=3)
    cv2.line(img, pt1=(0,h), pt2=(w,h), color=color_black, thickness=3)

def add_ball_to_frame(img):
    ball_source = './inputs/ball.png'
    ball_size = 2*radius
    pil_ball = Image.open(ball_source)
    back = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_back = Image.fromarray(back)
    pil_ball_resized = pil_ball.resize((ball_size,ball_size), Image.Resampling.BOX)
    pil_back.paste(pil_ball_resized, (rx_circle-radius,ry_circle-radius), pil_ball_resized)
    open_cv_back = np.array(pil_back) 
    img = cv2.cvtColor(open_cv_back, cv2.COLOR_BGR2RGB)
    return img

def add_text_to_frame(img):
    text_1 = f'{player_name_2}: {score_1}'
    text_2 = f'{player_name_1}: {score_2}'
    img = cv2.putText(img,text_1,org_1,font,font_scale,color_black,2,cv2.LINE_AA)
    img = cv2.putText(img,text_2,org_2,font,font_scale,color_black,2,cv2.LINE_AA)
    return img

def countdown_to_catch(starttime,hit,score_2,score_1,run_circle, img, player):
    endtime = time.time()
    timing = endtime-starttime
    if hit<= 10:
        diff = 10.00 - timing
    elif hit<=20 and hit>10:
        diff = 9.00 - timing
    elif hit<=30 and hit>20:
        diff = 8.00 - timing
    elif hit<=40 and hit>30:
        diff = 7.00 - timing
    elif hit<50 and hit>40:
        diff = 6.00 - timing

    if diff <= 0:
        if quadrant == 'first' or quadrant == 'fourth':
            score_2 += 1
        elif quadrant == 'second' or quadrant=='third':
            score_1 += 1
        run_circle=True
    
    countdown_time = str(diff)
    time_splitted = countdown_time.split(".")
    time_splitted[0] = int(time_splitted[0])
    time_splitted[1] = int (time_splitted[1][:2])
    timeformat = '{:02d}:{:02d}'.format(time_splitted[0], time_splitted[1])
    img = cv2.putText(img,timeformat,(282,25),font,0.85,color_red,2,cv2.LINE_AA)
    return score_1, score_2, run_circle, img, player

def game_over(hit, player_name_1, player_name_2):
    hit_ended = False
    if hit==50:
        text_1 = f'{player_name_2}: {score_1}'
        text_2 = f'{player_name_1}: {score_2}'
        if score_1 == score_2:
            text_3 = 'TIE!!!'
            res_color = color_yellow
            org_3 = (235,430)
        elif score_1>score_2:            
            text_3 = f'{player_name_2} WON!'
            res_color = color_green
            org_3 = (170,430)
        else:
            text_3 = f'{player_name_1} WON!'
            res_color = color_green
            org_3 = (170,430)

        hit_ended = True      
        res_font_scale = 2

        mixer.music.stop()
        mixer.music.load('./inputs/ta-da.mp3') 
        mixer.music.play()

        over = cv2.imread("./inputs/game_over.png")
        over = cv2.resize(over, (w,h))
        over = cv2.putText(over,text_1,org_1,font,font_scale,color_white,2,cv2.LINE_AA)
        over = cv2.putText(over,text_2,org_2,font,font_scale,color_white,2,cv2.LINE_AA)
        over = cv2.putText(over,text_3,org_3,font_bold,res_font_scale,res_color,2,cv2.LINE_AA)
        cv2.imshow('Volley Game', over)
    return hit_ended

def streaming(connection, address, package):
    while True:
        while len(package) < payload_size:
            package += connection.recv(4*1024) # 4*1024  - 4K
            if not package:
                cv2.destroyAllWindows()
                connection, address = s.accept()
                continue

        packed_msg_size = package[:payload_size]
        package = package[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(package) < msg_size:
            package += connection.recv(4*1024)
        frame_data = package[:msg_size]
        package = package[msg_size:]

        frame_local = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame_local = cv2.imdecode(frame_local, cv2.IMREAD_COLOR)

        return frame_local

def start_mouse_click(event, x, y, flags, param):
    if event ==cv2.EVENT_LBUTTONDOWN:
        global mp_drawing, mp_drawing_styles, mp_hands, hands
        global rx_circle,ry_circle
        global quadrant
        global starttime
        global run_circle
        global score_1, score_2
        global w,h
        global frame
        global data, data_2
        global payload_size
        global player
        global player_name_1, player_name_2
        global bonus 

        bonus = True
        conn,addr=s.accept()
        print(f"1. Got connection from:  {conn} : {addr}")

        conn_2,addr_2=s.accept()
        print(f"1. Got connection from:  {conn_2} : {addr_2}")

        if conn and conn_2: 
            payload_size = struct.calcsize(">L")

            back_source = './inputs/playing_field.jpg'
            background = cv2.imread(back_source)
            background = cv2.resize(background,(640,480))

            mixer.init() 
            mixer.music.load('./inputs/monkeys-spinning.mp3') 
            mixer.music.play(5)
            mixer.music.set_volume(0.01)

            rx_circle = random.randint(0+radius,640-radius)    
            ry_circle = random.randint(0+radius,480-radius)
            score_1, score_2 = 0, 0
            player = 0
            run_circle = True
            hit = 0
            quadrant = ''
            frame = None
            frame_1 = None
            frame_2 = None
            get_name_1 = True
            get_name_2 = True

            mp_drawing, mp_drawing_styles, mp_hands, hands = initialize_hand_tracking()

            while True:
                while get_name_1:
                    name_1 = conn.recv(4*1024)
                    player_name_1 = name_1.decode('utf-8')
                    player_name_1 = player_name_1[0:3]
                    player_name_1 = player_name_1.upper()
                    print(player_name_1)
                    if name_1:
                        get_name_1 = False
                        reply = 'Message from the server: ok'
                        conn.sendall(str.encode(reply))
                while get_name_2:
                    name_2 = conn_2.recv(4*1024)
                    player_name_2 = name_2.decode('utf-8')
                    player_name_2 = player_name_2[0:3]
                    player_name_2 = player_name_2.upper()
                    print('Next player: ', player_name_2)
                    if name_2:
                        if player_name_1 != player_name_2:
                            get_name_2 = False
                            reply = 'Message from the server: ok'
                            conn_2.sendall(str.encode(reply))
                        else:
                            reply = 'Message from the server: name_in_use'
                            conn_2.sendall(str.encode(reply))

                while len(data) < payload_size:
                    data += conn.recv(4*1024) # 4*1024  - 4K
                    if not data:
                        cv2.destroyAllWindows()
                        conn, addr = s.accept()
                        continue

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]
                while len(data) < msg_size:
                    data += conn.recv(4*1024)
                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame_1 = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                frame_1 = cv2.imdecode(frame_1, cv2.IMREAD_COLOR)


                while len(data_2) < payload_size:
                    data_2 += conn_2.recv(4*1024) # 4*1024  - 4K
                    if not data_2:
                        cv2.destroyAllWindows()
                        conn_2, addr_2 = s.accept()
                        continue

                packed_msg_size = data_2[:payload_size]
                data_2 = data_2[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]
                while len(data_2) < msg_size:
                    data_2 += conn_2.recv(4*1024)
                frame_data_2 = data_2[:msg_size]
                data_2 = data_2[msg_size:]

                frame_2 = pickle.loads(frame_data_2, fix_imports=True, encoding="bytes")
                frame_2 = cv2.imdecode(frame_2, cv2.IMREAD_COLOR)
  
                h = frame_1.shape[0]
                w = frame_1.shape[1]

                if run_circle:
                    run_circle = False
                    quadrant = find_quadrant(rx_circle,ry_circle)
                    rx_circle,ry_circle = ball_position()
                    starttime = time.time()
               
                if quadrant == 'first' or quadrant == 'fourth':
                    frame = frame_2
                    player = 2
                elif quadrant == 'second' or quadrant == 'third':
                    frame = frame_1
                    player = 1

                modified_frame, run_circle, hands, score_1, score_2, hit, bonus = hand_tracking(
                                                             frame, hands, hit, score_1, score_2, run_circle)

                initialize_net(modified_frame)
                initialize_border(modified_frame)
                    
                initialize_net(background)
                initialize_border(background)
                background = add_ball_to_frame(background)
                background = add_text_to_frame(background)
               
                    
                hit_ended = game_over(hit, player_name_1, player_name_2)
                if hit_ended:
                    break
                
                score_1, score_2, run_circle, background, player = countdown_to_catch(
                                                        starttime, hit, score_2, score_1, run_circle,  background, player)

                
                if player == 1:
                    cv2.imshow(f"Webcam of {player_name_1}: {addr}",  modified_frame)
                    cv2.imshow(f"Webcam of {player_name_2}: {addr_2}", frame_2)
                elif player == 2:
                    cv2.imshow(f"Webcam of {player_name_2}: {addr_2}",  modified_frame)
                    cv2.imshow(f"Webcam of {player_name_1}: {addr}", frame_1)

                cv2.imshow("Volley Game", background)

                background_new = cv2.imread(back_source)
                background = cv2.resize(background_new,(640,480))
    

                k = cv2.waitKey(10)
                if k == ord(' '):                                # pause: useful in the developing phase
                    while cv2.waitKey(10) == -1:
                        pass
                if k == 27:
                    mixer.music.stop()
                    cv2.destroyWindow(f"Webcam of {player_name_1}: {addr}")
                    cv2.destroyWindow(f"Webcam of {player_name_2}: {addr_2}")
                    break


if __name__ == "__main__" :

    start = cv2.imread('./inputs/start_game.jpeg')
    start = cv2.resize(start, (640,480))
    cv2.imshow('Volley Game', start)

    host_ip = '**.***.**.**'
    port = 1234
    socket_address = (host_ip, port)
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.bind(socket_address)
    s.listen(100)
    print("Listening at: ", socket_address)
    
    cv2.setMouseCallback('Volley Game', start_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
