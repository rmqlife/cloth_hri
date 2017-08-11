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
	cv_img = bridge.imgmsg_to_cv2(msg)
	_, hist = wrinkle.gabor_feat(cv_img)
	print hist
    motion = model.predict(hist)
    print motion

def listener():
	rospy.init_node('listener', anonymous = True)
	
	rospy.Subscriber('/camera/image/rgb_611205001943',Image, process_rgb)
	rospy.Subscriber('/yumi/ikSloverVel_controller/state', Float64MultiArray , callback)
    rospy.Subscriber('/yumi/ikSloverVel_controller/ee_cart_position', Float64MultiArray , callback)

	rospy.spin()

def test_move():
	

def prepare():
	global model
    model = regression.load()
    
if __name__ == '__main__':
    prepare()
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

