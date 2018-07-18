#! /usr/bin/env python

import rosbag

if __name__ == '__main__':
	bag = rosbag.Bag('c_2018-07-18-18-09-49.bag')
	#for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg']):
	#	print(topic, msg, t)
	x = [msg.ids for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
	print(x)
