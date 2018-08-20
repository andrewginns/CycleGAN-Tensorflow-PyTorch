#!/usr/bin/python           # This is server.py file

import socket               # Import socket module
import sys

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
print(host)
port = 7000                # Reserve a port for your service.
s.bind((host, port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
while True:
   c, addr = s.accept()     # Establish connection with client.
   print 'Got connection from', addr
   c.send('Thank you for connecting')

   data = c.recv(4096)
   print(sys.getsizeof(data))

   if not data: break
   c.sendall(data)
   c.close()                # Close the connection

   with open('C:/Users/ginns/Desktop/received.gz', 'w') as f:
       f.write(data)

   print("File received")

# import socket
# import struct
#
# HOST = socket.gethostname()
# PORT = 7000
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(5)
#
# print('SERVER STARTED RUNNING')
#
# while True:
#     client, address = s.accept()
#     print 'Got connection from', address
#     buf = ''
#     while len(buf) < 4:
#         buf += client.recv(4 - len(buf))
#     size = struct.unpack('!i', buf)[0]
#     with open('C:/Users/ginns/Desktop/received.gz', 'wb') as f:
#         while size > 0:
#             data = client.recv(1024)
#             f.write(data)
#             size -= len(data)
#     print('Data Saved')
#     client.sendall('Data Received')
#     client.close()