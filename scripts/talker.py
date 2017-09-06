#!/usr/bin/env python
# Software License Agreement (BSD License)
import sys
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
	print 'validate', max(abs(motion))
	toleration = 0.04
	if max(abs(motion))>toleration:
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

	data_name = sys.argv[1]
	# load data, to set the current position to target position  
	data = np.load(data_name)
	pos = data['pos']

	global have_current_pos; have_current_pos = False
	global have_im; have_im = False
	global current_pos,im
	# load the regression model
	model = regression.load_model(data_name)

	pub = rospy.Publisher('/yumi/ikSloverVel_controller/command', Float64MultiArray, queue_size=10)
	pub_im = rospy.Publisher('/cloth/wrinkle',Image,queue_size=10)

	rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , process_pos, queue_size = 2)
	rospy.Subscriber('/camera/image/rgb_611205001943',Image, process_rgb, queue_size=2)
	rospy.sleep(1)	

	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	
	vel = Float64MultiArray()
	vel.data = np.zeros(6)
	have_target = False
	while not rospy.is_shutdown():
		if not validate_pos(current_pos,pos):
			print "invalid position", current_pos
			vel.data = np.zeros(6)
			pub.publish(vel)
		elif have_im: 
			have_im = False
			if not have_target:
				avg, target_feat = wrinkle.gabor_feat(im,num_theta=8,grid=80)
				print target_feat
				have_target= True
			else: # have target_feat
				avg, hist = wrinkle.gabor_feat(im,num_theta=8,grid=80)
				bridge = CvBridge()
				pub_im.publish(bridge.cv2_to_imgmsg(avg,'mono8'))
				hist = np.array(target_feat) - np.array(hist)
				motion = model.predict(hist.reshape((1,-1))).ravel()
				motion = 0.3*motion
				motion = validate_motion(motion)
				vel.data = motion
				print "motion", motion
				pub.publish(vel)

		else: 
			print 'idle state'
			vel.data = np.zeros(6)
			pub.publish(vel)
		rate.sleep()
