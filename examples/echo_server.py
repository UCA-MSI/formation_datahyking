from socket import *

def echo_server(address):
    sock = socket(AF_INET, SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    sock.bind(address)
    sock.listen(5)
    while True:
        client, addr = sock.accept()  
        print("Connect", addr)
        echo_handler(client)


def echo_handler(client):
    while True:
        req = client.recv(100) 
        if not req:
            break
        client.send(req)    
    print("Closed")

if __name__ == "__main__":
    echo_server(('', 25000))

