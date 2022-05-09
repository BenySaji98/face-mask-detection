import RPi.GPIO as GPIO
import time

ir_sensor = 16
pump = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ir_sensor,GPIO.IN)
GPIO.setup(pump,GPIO.OUT)

def cleanup():
    GPIO.cleanup()


def drop_solution(HAND_WAIT=5,DROP_LENGTH=2):
    """
    drops solution for {DROP_LENGTH} seconds
    if hand/object is not found, wait for {HAND_WAIT} seconds
    returns true if solution was droped
    """
    start = time.time()
    while True:
        if GPIO.input(ir_sensor): # found hand/object
            print("IR sensor activated")
            break
        if time.time()-start >= HAND_WAIT: # wait timer expired
            return False

    print("Pump activated")
    GPIO.output(pump,True) # pump activated
    start = time.time()

    while GPIO.input(ir_sensor):
        if time.time() - start >= DROP_LENGTH: # timer expired
            print("maximum drop duration reached!")
            break
        time.sleep(0.2)
    else:
        print("IR sensor de-activated")

    print("Pump de-activated")
    GPIO.output(pump,False) # pump de-activated
    return True

if __name__ == '__main__':
    try:
        print("use CTRL-C for Keyboard Interrupt")
        while True:
            drop_solution()
            print("pausing activity for 5 seconds")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt")
    finally:
        print("Cleaning up")
        cleanup()
