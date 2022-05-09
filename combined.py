import drivers
from time import sleep
display = drivers.Lcd()
from smbus2 import SMBus
from mlx90614 import MLX90614
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)



while True:
    x = sensor.get_object_1()
        # Remember that your sentences can only be 16 characters long!
    print("Ambient Temperature :", sensor.get_ambient())
    print ("Object Temperature :", x)
    print("Writing to display")
    display.lcd_clear()
    display.lcd_display_string("TEMPERATURE:", 1)
    display.lcd_display_string(str(sensor.get_object_1()), 2)  # Write line of text to first line of display
    sleep(2)

    # Give time for the message to be read
        #display.lcd_clear()                                # Clear the display of any data
                                                   # Give time for the message to be read
#except KeyboardInterrupt:
    # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
    #print("Cleaning up!")
    #display.lcd_clear()
