import rospy
from led_msgs.srv import SetLEDs
from led_msgs.msg import LEDState

rospy.init_node('flight')

set_leds = rospy.ServiceProxy('led/set_leds', SetLEDs, persistent=True)

for i in range(72):
    if i <24:
        set_leds([LEDState(index=int(i), r=255, g=255, b=255)])
    elif i > 23 and i < 48:
        set_leds([LEDState(index=int(i), r=0, g=0, b=255)])
    if i > 47 :
        set_leds([LEDState(index=int(i), r=255, g=0, b=0)])
