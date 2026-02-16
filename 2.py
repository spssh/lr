import rospy
import pigpio
import time

rospy.init_node('flight')


pi = pigpio.pi()

pi.set_mode(13, pigpio.OUTPUT)

def sbros():

    pi.set_servo_pulsewidth(13, 1000)
    time.sleep(2)
    pi.set_servo_pulsewidth(13, 2000)

sbros()