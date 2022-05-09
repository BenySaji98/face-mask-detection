import RPi.GPIO as GPIO
import time

sensor = 16
pump = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(sensor,GPIO.IN)
GPIO.setup(pump,GPIO.OUT)

GPIO.output(pump,True)
print ("IR Sensor Ready.....")
print (" ")

try: 
   while True:
      if GPIO.input(sensor):
          GPIO.output(pump,False)
          print ("Object Detected")
          while GPIO.input(sensor):
              time.sleep(0.2)
      else:
          GPIO.output(pump,True)


except KeyboardInterrupt:
    GPIO.cleanup()