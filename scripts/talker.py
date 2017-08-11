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


def process_rgb(msg):
	bridge = CvBridge()
	global have_im; have_im = True
	global im; im = bridge.imgmsg_to_cv2(msg)

def process_pos(msg):
	global have_current_pos; have_current_pos = True
	global current_pos; current_pos = np.array(msg.data)

def predict(im, target_feat):
	avg, hist = wrinkle.gabor_feat(im,num_theta=8)
	hist = hist.reshape((1,-1))
	hist = target_feat - hist
	global model; motion = model.predict(hist)
	motion = motion.ravel()
	return motion

def prepare(target_eat):
	global model
	model = regression.load_model()
	# load an image to process a simple test
	im = cv2.imread('cloth.jpg')
	print 'prepare, predict(im):', predict(im, target_feat)

if __name__ == '__main__':
	# load data, to set the current position to target position  
	data = np.load('data2_grid.npz')
	pos = data['pos']
	feat = data['feat']

	init_pos = pos[1]
	reach_init = False
	target_feat = feat[1]
	global have_current_pos; have_current_pos = False
	global have_im; have_im = False
	global current_pos,im

	# initialize, set tt_motion to save the prev data
	tt_motion = np.array([])
	
	# load the regression model
	model = regression.load_model()
	# load an image for a simple test
	im = cv2.imread('cloth.jpg')
	avg, hist = wrinkle.gabor_feat(im,num_theta=8)
	hist = np.array(target_feat - hist)
	hist = hist.reshape((1,-1))

	motion = model.predict(hist).ravel()
	print 'test motion:', motion
	
	pub = rospy.Publisher('/yumi/ikSloverVel_controller/command', Float64MultiArray, queue_size=10)
	rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , process_pos)
	rospy.Subscriber('/camera/image/rgb_611205001943',Image, process_rgb)
	rospy.sleep(1)	

	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		vel = Float64MultiArray()
		# test the state machine
		# print "have_pos:", have_current_pos, "reach_init:", reach_init, "have_im:", have_im
		if not reach_init and have_current_pos:
			vel.data = 0.1*(init_pos-current_pos)
			print "reach init vel",vel.data
			if max(vel.data)<0.01:
				reach_init=True
				print "reach init state"
			print vel.data, init_pos,current_pos
			pub.publish(vel)
			have_current_pos = False

		elif reach_init and have_im:
			avg, hist = wrinkle.gabor_feat(im,num_theta=8)
			hist = np.array(target_feat - hist)
			motion = model.predict(hist.reshape((1,-1))).ravel()
			tt_motion = np.vstack((tt_motion,motion)) if tt_motion.size else motion
			vel.data = motion
			pub.publish(vel)
			print "motion", motion
			have_im = False
		rate.sleep()
