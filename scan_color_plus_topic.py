#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from clover.srv import SetLEDEffect

# Инициализация ROS-ноды и моста OpenCV
rospy.init_node('color_scanner')
bridge = CvBridge()
image_pub = rospy.Publisher('/color_scanner/debug', Image, queue_size=1)
result_pub = rospy.Publisher('/color_scanner/result', String, queue_size=1)

# Инициализация сервиса управления светодиодной лентой
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

# Параметры сканирования (в пикселях от центра изображения)
SCAN_RADIUS_PX = 150

# Определение целевых цветов в HSV + RGB для LED
TARGET_COLORS = {
    'red': ([0, 100, 100], [10, 255, 255], (255, 0, 0)),
    'green': ([40, 50, 50], [80, 255, 255], (0, 255, 0)),
    'blue': ([100, 100, 50], [140, 255, 255], (0, 0, 255))
}

def detect_shape(contour):
    """Определяет тип фигуры по количеству углов"""
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    sides = len(approx)
    if sides == 3:
        return 'triangle'
    elif sides == 4:
        return 'square'
    elif sides == 5:
        return 'pentagon'
    elif sides == 6:
        return 'shestiugolnik'
    else:
        return 'krug'

def detect_dominant_color(roi):
    """Определяет доминирующий цвет из заданных в ROI"""
    if roi.size == 0:
        return None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    max_area = 0
    dominant_color = None
    dominant_rgb = None

    for color_name, (lower, upper, rgb) in TARGET_COLORS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        area = cv2.countNonZero(mask)
        if area > max_area:
            max_area = area
            dominant_color = color_name
            dominant_rgb = rgb

    return (dominant_color, dominant_rgb) if max_area > 50 else (None, None)

def image_callback(msg):
    """Обработка кадров с камеры — работает постоянно"""
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Маска круга под дроном
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), SCAN_RADIUS_PX, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Поиск контуров
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_color = None
    detected_shape = None
    detected_rgb = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            roi = img[y:y+h_cnt, x:x+w_cnt]

            color, rgb = detect_dominant_color(roi)
            shape = detect_shape(cnt)

            if color:
                detected_color = color
                detected_shape = shape
                detected_rgb = rgb

                # Рисуем контур **фиолетовым** цветом
                cv2.drawContours(img, [cnt], -1, (255, 0, 255), 2)

                # Подписываем над объектом
                label = f"{color} {shape}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Вывод в левый верхний угол и управление LED
    if detected_color and detected_shape:
        result_text = f"{detected_color} {detected_shape}"
        cv2.putText(img, f"Detected: {result_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        result_pub.publish(result_text)
        if detected_rgb:
            set_effect(r=detected_rgb[2], g=detected_rgb[1], b=detected_rgb[0], effect='fill')
    else:
        # Если нет объекта — гасим LED
        set_effect(r=0, g=0, b=0, effect='fill')

    image_pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))

# Подписка на топик с камерой
image_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback, queue_size=1)

def scan_color_and_type():
    """Функция сканирования — вызывается во время облёта"""
    rospy.loginfo("Starting color and shape scan...")
    try:
        result_msg = rospy.wait_for_message('/color_scanner/result', String, timeout=3.0)
        rospy.loginfo(f"Scanned result: {result_msg.data}")
        return result_msg.data
    except rospy.ROSException:
        rospy.logwarn("No object detected during scan")
        set_effect(r=0, g=0, b=0, effect='fill')
        return None

def perform_flight_route():
    """Пример функции облёта с вызовом сканирования"""
    rospy.loginfo("Starting flight route...")

    from clover.srv import Navigate
    rospy.wait_for_service('navigate')
    navigate = rospy.ServiceProxy('navigate', Navigate)

    # Взлёт
    rospy.loginfo("Taking off...")
    navigate(x=0, y=0, z=0.75, frame_id='body', auto_arm=True)
    rospy.sleep(5)

    # Точка A
    rospy.loginfo("Flying to point A...")
    navigate(x=1, y=1, z=0.75, frame_id='aruco_map')
    rospy.sleep(3)
    scan_color_and_type()

    # Точка B
    rospy.loginfo("Flying to point B...")
    navigate(x=1, y=0, z=0.75, frame_id='aruco_map')
    rospy.sleep(3)


    # Посадка
    rospy.loginfo("Landing...")
    rospy.wait_for_service('land')
    land = rospy.ServiceProxy('land', Trigger)
    land()
    set_effect(r=0, g=0, b=0, effect='fill')
    rospy.loginfo("Mission complete")

if __name__ == '__main__':
    try:
        perform_flight_route()
    except rospy.ROSInterruptException:
        set_effect(r=0, g=0, b=0, effect='fill')
        pass