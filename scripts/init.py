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

def process_pos(msg):
	global have_current_pos; have_current_pos = True
	global current_pos; current_pos = np.array(msg.data)

def validate_motion(motion):
	print 'validate', max(motion)
	toleration = 0.04
	if max(motion)>toleration:
		motion = motion * toleration/max(motion)
	return motion

if __name__ == '__main__':
	data_name = 'data2.npz'
	data = np.load(data_name)
	pos = data['pos']
	feat = data['feat']

	init_pos = pos[1]
	reach_init = False

	global have_current_pos; have_current_pos = False

	global current_pos,im

	pub = rospy.Publisher('/yumi/ikSloverVel_controller/command', Float64MultiArray, queue_size=10)
	rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , process_pos)

	rospy.sleep(1)	

	rospy.init_node('init', anonymous=True)
	rate = rospy.Rate(10) # 10hz

	vel = Float64MultiArray()
	while not rospy.is_shutdown():
		if not reach_init and have_current_pos:
			if max(init_pos-current_pos)<0.01:
				reach_init=True
				print "reach init state"
				motion = np.zeros(6)
			else:
				motion = 0.1*(init_pos-current_pos)
				motion = validate_motion(motion)
				print "reaching init vel",motion
			vel.data = motion
			pub.publish(vel)
			have_current_pos = False
		else: 
			print 'idle state'
			vel.data = np.zeros(6)
			pub.publish(vel)
		rate.sleep()
