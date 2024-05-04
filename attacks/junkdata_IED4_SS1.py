import random
import socket
size = 1024
select = ''
for _ in range(len(72)):
operate = ''
for _ in range(len(72)):
	operate += '\x' + hex(rand.int(0,15)) + hex(rand.int(0,15))
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(('6.87.151.210',20000))
s.send(str(select))
data = s.recv(size)
s.send(operate)
data = s.recv(size)
print 'IED-4 Tripped'
s.close()
