#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import cv2
import wrinkle
import regression

def callback(msg):
	print np.array(msg.data)

def process_rgb(msg):
	bridge = CvBridge()
	im = bridge.imgmsg_to_cv2(msg)
	motion = predict(im)
	print motion

def listener():
	rospy.init_node('listener', anonymous = True)
	rospy.Subscriber('/camera/image/rgb_611205001943',Image, process_rgb)
	rospy.Subscriber('/yumi/ikSloverVel_controller/state', Float64MultiArray , callback)
	rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , callback)
	rospy.spin()	

def predict(im):
	avg, hist = wrinkle.gabor_feat(im,num_theta=8)
	hist = np.array(hist)
	hist = hist.reshape((1,-1))
	motion = model.predict(hist)
	motion = motion.ravel()
	global tt_motion
	tt_motion = np.vstack((tt_motion,motion)) if tt_motion.size else motion
	return motion

def prepare():
	global model
	model = regression.load_model()
	# load an image to process a simple test
	im = cv2.imread('cloth.jpg')
	print predict(im)


if __name__ == '__main__':
	global tt_motion
	tt_motion = np.array([])
	prepare()
	try:
		listener()
	except rospy.ROSInterruptException:
		np.save('motion',tt_motion)
		pass

