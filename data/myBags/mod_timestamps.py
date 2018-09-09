import sys, rosbag

with rosbag.Bag(sys.argv[2], 'w') as outbag:
    for topic, msg, t in rosbag.Bag(sys.argv[1]).read_messages():
        # This also replaces tf timestamps under the assumption 
        # that all transforms in the message share the same timestamp
        if topic == "/tf" and msg.transforms:
            outbag.write(topic, msg, msg.transforms[0].header.stamp)
        else:
            outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)