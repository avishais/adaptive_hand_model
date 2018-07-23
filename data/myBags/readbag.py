#! /usr/bin/env python

import rosbag
import rospy
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()                                               

parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

# -----------------------------------------

# Naive NN search while assuming a sorted list
def NearestNeighbor(X, x):
	ix = 0
	while x > X[ix]:
		ix += 1

	if abs(x - X[ix]) < abs(x - X[ix-1]):
		return ix
	else:
		return ix-1

# ------------------------------------------


bag = rosbag.Bag(args.file)
str = args.file[:-4]
# for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg']):
# 	print(topic, msg, t)

Tt = [t for (topic, msg, t) in bag.read_messages(topics=['/gripper_t42/vel_ref_monitor'])]
T = [t for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
print len(T), len(Tt)

t_init = np.min([Tt[0].to_sec(), T[0].to_sec()])
print('Vel. Ref.,  Object')
for i in range(0,30):
	print('%9.3f, %7.3f' % (Tt[i].to_sec()-t_init, T[i].to_sec()-t_init))

exit(1)

##### Get markers
ids = [msg.ids for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
posx = [msg.posx for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
posy = [msg.posy for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
angles = [msg.angles for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
T = [t for (topic, msg, t) in bag.read_messages(topics=['/marker_tracker/image_space_pose_msg'])]
# ids=ids[1:]
# posx=posx[1:]
# posy=posy[1:]
# angles=angles[1:]
# T=T[1:]

# Time
n = len(T)
T_init = T[0].to_sec()
Ts = [None]*n
for i in range(len(T)):
	Ts[i] = T[i].to_sec()
Ts[:] = [x - T_init for x in Ts]
Ts = np.array(Ts)

# Base - marker 0
m = 0
pos_base = []
for i in range(n):
	if m in ids[i]:
		j = ids[i].index(m)
		pos_base.append([posx[i][j], posy[i][j], angles[i][j]])
	else:
		pos_base.append([-1, -1, -1])
pos_base = np.array(pos_base)

# Left distal finger - marker 1
m = 1
pos_leftDistal = []
for i in range(n):
	if m in ids[i]:
		j = ids[i].index(m)
		pos_leftDistal.append([posx[i][j], posy[i][j], angles[i][j]])
	else:
		pos_leftDistal.append([-1, -1, -1])
pos_leftDistal = np.array(pos_leftDistal)

# Right distal finger - marker 2
m = 2
pos_rightDistal = []
for i in range(n):
	if m in ids[i]:
		j = ids[i].index(m)
		pos_rightDistal.append([posx[i][j], posy[i][j], angles[i][j]])
	else:
		pos_rightDistal.append([-1, -1, -1])
pos_rightDistal = np.array(pos_rightDistal)

# left proximal finger - marker 3
m = 3
pos_leftProx = []
for i in range(n):
	if m in ids[i]:
		j = ids[i].index(m)
		pos_leftProx.append([posx[i][j], posy[i][j], angles[i][j]])
	else:
		pos_leftProx.append([-1, -1, -1])
pos_leftProx = np.array(pos_leftProx)

# Right proximal finger - marker 4
m = 4
pos_rightProx = []
for i in range(n):
	if m in ids[i]:
		j = ids[i].index(m)
		pos_rightProx.append([posx[i][j], posy[i][j], angles[i][j]])
	else:
		pos_rightProx.append([-1, -1, -1])
pos_rightProx = np.array(pos_rightProx)

# Object - Marker 5
m = 5
pos_object = []
for i in range(n):
	if m in ids[i]:
		j = ids[i].index(m)
		pos_object.append([posx[i][j], posy[i][j], angles[i][j]])
	else:
		pos_object.append([-1, -1, -1])
pos_object = np.array(pos_object)

##### Get gripper data

vel_action = [msg.data for (topic, msg, t) in bag.read_messages(topics=['/gripper_t42/vel_ref_monitor'])]
vel_action = np.array(vel_action)

pos_gripper = [msg.data for (topic, msg, t) in bag.read_messages(topics=['/gripper/pos'])]
pos_gripper = np.array(pos_gripper)
# print pos_gripper

Tt = [t for (topic, msg, t) in bag.read_messages(topics=['/gripper_t42/vel_ref_monitor'])]
Tv = [None]*len(Tt)
for i in range(len(Tt)):
	Tv[i] = Tt[i].to_sec()#-T_init
# Tv = np.array(Tv)

# print len(Tv), len(Ts)
# for i in range(0,10):
# 	print Tv[i]-Tv[0], T[i].to_sec()-Tv[0]
# print Tv[0:10], Ts[0:10]


# Tv[:] = [x - T_init for x in Tv]
# Tv = np.array(Tv)
# # print (Tv)#, len(T2)
# ix = NearestNeighbor(Ts, Tv[0])
# # print Ts[ix], Ts[ix-1], Tv[0]
# print ix, len(Ts), len(Tv)
# Tvn = [0]*len(Ts)
# for i in range(ix, ix+len(Tv)):
# 	# print i, i+ix
# 	Tvn[i]=Tv[i-ix]
# for t1, t2 in zip(Ts[250:600], Tvn[250:600]):
# 	print t1, t2

##### write data to file
# D = np.concatenate((pos_base, pos_object), axis=1) #, vel_action)
# print D.shape, vel_action.shape

# D = np.concatenate((D, vel_action), axis=1) #, vel_action)
# D = np.concatenate((Ts[:,None], D), axis=1) 

# np.savetxt(str + '.db', D, fmt='%.3e')#


##### Plots
# plt.plot(pos_object[:,0], pos_object[:,1],'.-')
# plt.plot(pos_base[:,0], pos_base[:,1])
# plt.plot(pos_leftDistal[:,0], pos_leftDistal[:,1])
# plt.plot(pos_rightDistal[:,0], pos_rightDistal[:,1])

# plt.show()


