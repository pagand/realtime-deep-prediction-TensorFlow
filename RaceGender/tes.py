from random import random
import threading
import time

Result = None

def background_calculation(inpu1,inpu):
    # here goes some long calculation
    time.sleep(5)

    # when the calculation is done, the result is stored in a global variable
    global Result
    Result = inpu

def main():

    global Result
    threads = []

    for i in range(12):
        if i % 3 == 0:
            thread = threading.Thread(target=background_calculation, args=('hi',i,))
            threads.append(thread)
            thread.start()
        if Result:
            print('The result is Ready', Result)
            Result = None
        print(i)
        time.sleep(1)

if __name__ == '__main__':
    main()