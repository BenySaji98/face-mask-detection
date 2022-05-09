from time import sleep
from drivers import Lcd
from smbus2 import SMBus
from mlx90614 import MLX90614

bus = SMBus(1)
display = Lcd() # 16x2 LCD display module
temp_sensor = MLX90614(bus, address=0x5A)


try:
    while True:
        obj_temp = temp_sensor.get_object_1()
        amb_temp = temp_sensor.get_ambient()
        print(f"Ambient Temperature : {amb_temp}")
        print(f"Object Temperature  : {obj_temp}")
        print("Writing to display")
        display.lcd_clear()
        display.lcd_display_string("TEMPERATURE:", 1)
        display.lcd_display_string(str(obj_temp), 2)

        sleep(2)
        display.lcd_clear()

except KeyboardInterrupt:
    print("Cleaning up!")
    display.lcd_clear()
