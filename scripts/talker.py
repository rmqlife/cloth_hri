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


def validate_motion(motion):
	print 'validate', max(motion)
	toleration = 0.04
	if max(motion)>toleration:
		print "bad"
		motion = np.zeros(6)
	return motion


def validate_pos(current_pos,pos):
    upper = np.zeros(6)
    lower = np.zeros(6)
    for i in range(pos.shape[1]):
        upper[i] = max(pos[:,i])+0.05
        lower[i] = min(pos[:,i])-0.05
    for i in range(len(current_pos)):
        if current_pos[i]<lower[i]:
            print 'lower',lower
            return False
        if current_pos[i]>upper[i]:
            print 'upper', upper
            return False
    return True
	

if __name__ == '__main__':

	data_name = 'data2.npz'
	# load data, to set the current position to target position  
	data = np.load(data_name)
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
	model = regression.load_model(data_name)
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
	
	online_move = True

	vel = Float64MultiArray()
	while not rospy.is_shutdown():

		# test the state machine
		# print "have_pos:", have_current_pos, "reach_init:", reach_init, "have_im:", have_im
		if not reach_init and have_current_pos:
			motion = 0.1*(init_pos-current_pos)
			motion = validate_motion(motion)
			print "reaching init vel",motion
			if max(motion)<0.001:
				reach_init=True
				print "reach init state"
				motion = np.zeros(6)
			vel.data = motion
			pub.publish(vel)
			have_current_pos = False
		elif not validate_pos(current_pos,pos):
			print "invalid position", current_pos
			vel.data = np.zeros(6)
			pub.publish(vel)

		elif reach_init and have_im and online_move: 
			avg, hist = wrinkle.gabor_feat(im,num_theta=8)
			hist = np.array(target_feat - hist)
			motion = model.predict(hist.reshape((1,-1))).ravel()
			motion = 0.1*motion
			motion = validate_motion(motion)
			vel.data = motion
			pub.publish(vel)
			print "motion", motion
			have_im = False
		else: 
			print 'idle state'
			#vel.data = np.zeros(6)
			pub.publish(vel)
		rate.sleep()
