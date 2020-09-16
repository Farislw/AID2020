"""
from socket import*
sockfd = socket()
server_addr = ('127.0.0.1',8888)
sockfd.connect(server_addr)
while True:
    data = input("msg >>")
    if not data:
        break
    sockfd.send(data.encode())
    data = sockfd.recv(1024)
    print("server:",data.decode())
sockfd.close()
"""
from socket import *
s = socket()
s.connect(('127.0.0.1',8888))
f = open('test','rb')
while True:
    data = f.read(1024)
    if not data:
        break
    s.send(data)
f.close()
s.close()
