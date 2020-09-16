from socket import *
import struct
st = struct.Struct('i32sif')
s = socket(AF_INET,SOCK_DGRAM)
s.bind(('127.0.0.1',8888))
f = open('student.txt','a')
while True:
    data,addr = s.recvfrom(1024)
    data = st.unpack(data)
    info = "%d %-10s %d %.1f\n"%data
    f.write(info)
    f.flush()
