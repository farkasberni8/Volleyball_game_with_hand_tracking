import cv2
import socket
import struct
import pickle

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '**.***.**.**'  
host_port = 1234
client_socket.connect((host_ip, host_port))


cam = cv2.VideoCapture(0)
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),95]
frame_count = 0

send_name = True

while True:
    if send_name:
        name = input('Type your name (3 characters): ')
        client_socket.send(str.encode(name))
        response = client_socket.recv(4*1024)
        response = response.decode('utf-8')
        print(response)
        send_name = False
    if response == 'Message from the server: name_in_use':
        name = input('The name is already in use, choose another one (3 characters): ')
        client_socket.send(str.encode(name))
        response = client_socket.recv(4*1024)
        response = response.decode('utf-8')
        print(response)
        send_name = False
    if response == 'Message from the server: ok':
        ret, frame = cam.read()  
        frame = cv2.flip(frame,180)
        result, image = cv2.imencode('.jpg', frame, encode_param)

        if frame_count % 8 == 0:
            data = pickle.dumps(image, 0)
            size = len(data)
            client_socket.sendall(struct.pack(">L", size) + data)
            cv2.imshow('client',frame)
            
        frame_count += 1
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    
cam.release()
cv2.destroyAllWindows()
