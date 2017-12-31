import http.server
import socketserver

class ServerHTTP:
    
    def __init__(self, ip="", port=9876):
        self.handler = httpServerHandler
        self.httpd = socketserver.TCPServer((ip, port), self.handler)
        # self.httpd.serve_forever()

    def receive(self):
        self.httpd._handle_request_noblock()
        print(dir(self.httpd.RequestHandlerClass))
        return self.httpd.RequestHandlerClass.received

    def send(self, message):
        self.httpd.response = message
        self.httpd._handle_request_noblock()



class httpServerHandler(http.server.BaseHTTPRequestHandler):
    """
    A simple HTTP server capable of handling GET and POST requests
    """

    def __init__(self):
        self.received = b''

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.received = None
        
    def _set_headers(self, response=None, connection=None):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')

        if response is not None:
            self.send_header("Content-Length", len(response))
        if connection is not None:
            self.send_header('Connection', connection)
        self.end_headers()

    def do_GET(self):
        self.protocol_version = "HTTP/1.0"
        response = self.response
        self._set_headers(response=response)
        self.wfile.write(response)

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        self.protocol_version = "HTTP/1.1"
        content_length = int(self.headers['Content-Length'])

        payload = self.rfile.read(content_length)
        print("Post", payload)
        self.response = ""
        self.received = payload
        self.do_RESPONSE('ack')
    def receive(self):
        print('receive')
        return self.received

    def send(self, message):
        self.do_RESPONSE(message)
    
    def do_RESPONSE(self, response):
        print('responding', response)
        self._set_headers(response=response)
        self.wfile.write(response)
        self._set_response()


def resp_func(self, payload):
    if payload.startswith(b'screenshot'):
        return(b"ack")
    else:
        return(b"wudder")

if __name__ == "__main__":
    print("hello?!")
    handler = httpServerHandler
    handler.response_function = resp_func
    PORT = 9876
    httpd = socketserver.TCPServer(("", PORT), handler)
    print(dir(httpd.RequestHandlerClass))
    print("serving at port", PORT)
    httpd.serve_forever()

