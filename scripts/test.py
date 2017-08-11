#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Float64MultiArray

def callback(data):
	print 'hehe',
	print data

def listener():
	rospy.init_node('listener', anonymous = True)
	rospy.Subscriber('/yumi/ikslovervel_controller/state', Float64MultiArray , callback)

	rospy.spin()

if __name__ == '__main__':
	listener()

