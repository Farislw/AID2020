from socket import *
from select import *
class HTTPServer:
    def __init__(self,host = '0.0.0.0',port = 8000,dir = None):
        self.host = host
        self.port = port
        self.dir = dir
        self.address = (host,port)
        self.create_socket()
        self.bind()
        self.rlist =[]
        self.wlist =[]
        self.xlist = []
    def create_socket(self):
        self.sockfd = socket()
        self.sockfd.setsockopt(SOL_SOCKET,SO_REUSEADDR,1)
    def bind(self):
        self.sockfd.bind(self.address)
    def server_forever(self):
        self.sockfd.listen(3)
        print("listen the port %d"%self.port)
        self.rlist.append(self.sockfd)
        while True:
            rs,wx,xs = select(self.rlist,self.wlist,self.xlist)
            for r in rs:
                if r is self.sockfd:
                    c,addr = r.accept()
                    print("connect from",addr)
                    self.rlist.append(c)
                else:
                    self.handle(r)
    def handle(self,connfd):
        request = connfd.recv(4096)
        if not request:
            self.rlist.remove(connfd)
            connfd.close()
            return
        request_line = request.splitlines()[0]
        info = request_line.decode().split(' ')[1]
        print(connfd.getpeername(),":",info)
        if info == '/' or info[-5:] == '.html':
            self.get_html(connfd,info)
        else:
            self.get_data(connfd,info)
    def get_html(self,connfd,info):
        if info == '/':
            # request index
            filename = self.dir +"/index.html"
        else:
            filename = self.dir + info
        try:
            fd = open(filename)
        except Exception:
            #not exist
            response = "HTTP/1.1 404 Not Found\r\n"
            response += "Content-Type:TEXT/HTML\r\n"
            response += '\r\n'
            response += '<h1>Sorry....</h1>'
        else:
            response = "HTTP/1.1 200 ok\r\n"
            response += "Content-Type:TEXT/HTML\r\n"
            response += '\r\n'
            response += fd.read()
        finally:
            #jiang xiangying fa song gei liu lanqi
            connfd.send(response.decode())
    def get_data(self,connfd,info):
        pass



if __name__ == "__main__":
    """
    by class HTTPSERVER
    """
    HOST = '0.0.0.0'
    PORT = 8000
    DIR = './static'
    httpd = HTTPServer(HOST,PORT,DIR)
    httpd.server_forever()
