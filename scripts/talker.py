#!/usr/bin/env python
# Software License Agreement (BSD License)

import rospy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import cv2
import wrinkle
import regression

def talker():
	pub = rospy.Publisher('/yumi/ikSloverVel_controller/command', Float64MultiArray, queue_size=10)
	rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , callback)
	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		vel = Float64MultiArray()
		vel.data = 0.1*(target_pos-current_pos)
		print vel.data, target_pos,current_pos
		pub.publish(vel)
		rate.sleep()


def callback(msg):
	global current_pos
	current_pos = np.array(msg.data)

def listener():
	rospy.init_node('listener', anonymous = True)
	# rospy.Subscriber('/yumi/ikSloverVel_controller/state', Float64MultiArray , callback)
	rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , callback)
	rospy.spin()

if __name__ == '__main__':
	data = np.load('/home/jjhu/catkin_ws/src/cloth_hri/scripts/data2_grid.npz')
	pos = data['pos']
	global target_pos
	target_pos = pos[1]
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
