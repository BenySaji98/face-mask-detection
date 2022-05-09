from time import sleep
from drivers import Lcd
from smbus2 import SMBus
from mlx90614 import MLX90614

bus = SMBus(1)
display = Lcd() # 16x2 LCD display module
temp_sensor = MLX90614(bus, address=0x5A)

def cleanup():
    global display
    display.lcd_clear()

def scan_temp_and_display():
    """
    scans the Temperature and returns it. Also displays into LCD driver
    """

    obj_temp = temp_sensor.get_object_1()
    amb_temp = temp_sensor.get_ambient()

    print(f"Ambient Temperature : {amb_temp}")
    print(f"Object Temperature  : {obj_temp}")

    print("Writing to display")
    display.lcd_clear()
    display.lcd_display_string("TEMPERATURE:", 1)
    display.lcd_display_string(str(obj_temp), 2)

    return obj_temp

if __name__ == '__main__':
    try:
        print("use CTRL-C for Keyboard Interrupt")
        while True:
            scan_temp_and_display()
            print("pausing activity for 5 seconds")
            sleep(5)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt")
    finally:
        print("Cleaning up")
        cleanup()
