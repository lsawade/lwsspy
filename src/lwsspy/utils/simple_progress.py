import time

counter = 0
width = 50


for x in range(0, 1000):

    if x % round(1000/50) == 0:
        counter += 1
        end = '\n' if counter == width else '\r'
        print('[' + counter * '■' + (width-counter) * '-' + ']', end=end)
    time.sleep(0.01)

print('Hello')
