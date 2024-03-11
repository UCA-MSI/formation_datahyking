from threading import Thread
import time

value = 0

def update():
    global value
    tmp = value
    time.sleep(.001)
    value = tmp + 10
    
threads = []

for i in range(20):
    threads.append(Thread(target=update))

for i in range(20):
    threads[i].start()

for i in range(20):
    threads[i].join()

print(value)