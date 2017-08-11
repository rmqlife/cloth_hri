#!/usr/bin/env python
# Software License Agreement (BSD License)

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

def talker():
    pub = rospy.Publisher('/yumi/ikSloverVel_controller/command', Float64MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        vel = [0.01,0.01,0.01,0.01,0.01,0.01]
        print vel
        pub.publish(vel)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
