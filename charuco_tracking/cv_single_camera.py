#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (C) 2019 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
#|    Authors:  Konstantinos Chatzilygeroudis (maintainer)
#|              Fanjun Bu
#|              Bernardo Fichera
#|    email:    konstantinos.chatzilygeroudis@epfl.ch
#|              frankbu0616@gmail.com
#|              bernardo.fichera@epfl.ch
#|    website:  lasa.epfl.ch
#|
#|    You can redistribute it and/or modify
#|    it under the terms of the GNU General Public License as published by
#|    the Free Software Foundation, either version 3 of the License, or
#|    (at your option) any later version.
#|
#|    This code is distributed in the hope that it will be useful,
#|    but WITHOUT ANY WARRANTY; without even the implied warranty of
#|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#|    GNU General Public License for more details.
#|
# System libs
import glob
import sys
import os
import copy
import argparse

# Numeric libs
import cv2
import numpy as np
from numpy import linalg as LA

# Helper functions
import cv2
from cv2 import aruco
from help_functions import transform_from_board_to_camera, transform_from_base_to_camera, \
    compute_board_origin_in_base_frame, draw_axis, compute_center_from_three_points, compute_center_from_four_points, \
    apply_filter, apply_filter_watch, compute_transformation_from_board_to_base, construct_new_axis_robot

# ROS
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf
# from geometry_msgs.msg import PointStamped
# from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError

class Tracker:
    def __init__(self):
        self.fr1 = 0
        self.fr2 = 0
        self.cam1_state = "wait"
        self.cam2_state = "wait"

        self.cvBridge = CvBridge()

        self.got_dims = False

        self.cam1 = dict.fromkeys(['rgb'])
        self.cam2 = dict.fromkeys(['rgb'])

        self.cam1_topic = 		"/camera/color/image_raw"
        # self.cam1_topic = 		"/camera1/color/image_raw"
        # self.cam1_ir1_topic = 	"/camera1/infra1/image_rect_raw"
        # self.cam1_ir2_topic = 	"/camera1/infra2/image_rect_raw"


        self.cam2_topic = 		"/camera2/color/image_raw"
        # self.cam2_ir1_topic = 	"/camera2/infra1/image_rect_raw"
        # self.cam2_ir2_topic = 	"/camera2/infra2/image_rect_raw"

        self.pub_msg = PoseStamped()
        self.pub_msg.header = Header()
        self.pub2_msg = PoseStamped()
        self.pub2_msg.header = Header()
        self.pub_watch_msg = PoseStamped()
        self.pub_watch_msg.header = Header()

        self.br = tf.TransformBroadcaster()

    def cam1_callback(self, data):
        # if self.cam1_state == "ready" and self.cam2_state == "wait":
        #     return
        self.cam1['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")
        self.cam1_state = "ready"
        self.fr1 += 1

    def cam2_callback(self, data):
        if self.cam2_state == "ready" and self.cam1_state == "wait":
            return
        self.cam2['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")
        self.cam2_state = "ready"
        self.fr2 += 1

    def init_ros_topics(self):
        # ROS topic: retrieve frames
        rospy.Subscriber(self.cam1_topic, Image, self.cam1_callback)
        # rospy.Subscriber(self.cam2_topic, Image, self.cam2_callback)

        self.pub = rospy.Publisher('/board_to_robot', PoseStamped, queue_size=10)
        self.pub2 = rospy.Publisher('/board_to_robot_left', PoseStamped, queue_size=10)
        self.pub_watch = rospy.Publisher('/board_to_watch', PoseStamped, queue_size=10)

    def run(self):
        # Initialise subscribing topics
        self.init_ros_topics()

        # Wait for cameras to be ready before going ahead
        while ((self.cam1_state != "ready") and (not rospy.is_shutdown())): # or (self.cam2_state != "ready")):
            continue

        if rospy.is_shutdown():
            return

        # load camera calibration info from cameras
        calib = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        # calib2 = rospy.wait_for_message('/camera2/color/camera_info', CameraInfo)

        mtx = np.reshape(calib.K, [3,3])
        dist = np.array([calib.D])

        # marker size in meters.
        board_square_length = 0.032 # 0.042
        board_marker_size = 0.0265 # 0.034
        robot_square_length = 0.161
        robot_marker_size = 0.140
        robot_left_square_length = 0.134
        robot_left_marker_size = 0.104
        watch_square_length = 0.07 #0.0595
        watch_marker_size = 0.0535#0.0455#0.043

        # robot on the right
        # initialize array to store location data for base markers in the camera's frame.
        base_marker1 = np.array([0,0,0])
        base_marker2 = np.array([0,0,0])
        base_marker3 = np.array([0,0,0])

        # robot on the left
        #initialize location data for base markers in the camera's frame.
        base_marker_l1 = np.array([0,0,0])
        base_marker_l2 = np.array([0,0,0])
        base_marker_l3 = np.array([0,0,0])

        # watch
        #initialize location data for base markers in the camera's frame.
        watch_base_marker1 = np.array([0,0,0])
        watch_base_marker2 = np.array([0,0,0])
        watch_base_marker3 = np.array([0,0,0])
        watch_base_marker4 = np.array([0,0,0])

        # initilaize location data for charuco board
        board_origin = np.array([0,0,0])

        # defining different dictionaries to use for different coordinate systems
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
        aruco_dict_robot_l = aruco.Dictionary_get(aruco.DICT_6X6_250)
        aruco_dict_robot_r = aruco.Dictionary_get(aruco.DICT_4X4_250)
        aruco_dict_watch = aruco.Dictionary_get(aruco.DICT_7X7_250)

        board = aruco.CharucoBoard_create(9, 6, board_square_length, board_marker_size, aruco_dict)
        charuco_robot_r = aruco.CharucoBoard_create(3, 3, robot_square_length, robot_marker_size, aruco_dict_robot_r)
        charuco_robot_l = aruco.CharucoBoard_create(3, 3, robot_left_square_length, robot_left_marker_size, aruco_dict_robot_l)
        charuco_watch = aruco.CharucoBoard_create(3, 3, watch_square_length, watch_marker_size, aruco_dict_watch)
        parameters = aruco.DetectorParameters_create()

        # filter
        a_filter = 0.95

        # Initialization of variables
        Tb_r = np.identity(4)
        rot_mat = np.identity(3)
        robot_origin = np.array([[0, 0, 0]])
        Tb_l = np.identity(4)
        rot_mat_l = np.identity(3)
        robot_origin_l = np.array([[0, 0, 0]])
        Tb_w = np.identity(4)
        rot_mat_w = np.identity(3)
        watch_origin = np.array([[0, 0, 0]])

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            QueryImg = copy.deepcopy(self.cam1['rgb'])
            # cv2.imwrite('./out/imgA_{}.png'.format(self.fr1), QueryImg)

            # grayscale image
            gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

            # Detect Aruco markers, _r refers to robot on the right, _l referes to robot on the left.
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            corners_r, ids_r, rejectedImgPoints_r = aruco.detectMarkers(gray, aruco_dict_robot_r, parameters=parameters)
            corners_l, ids_l, rejectedImgPoints_l = aruco.detectMarkers(gray, aruco_dict_robot_l,  parameters=parameters)

            watch_corners, watch_ids, watch_rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict_watch,  parameters=parameters)

            # Refine detected markers
            # Eliminates markers not part of our board, adds missing markers to the board
            corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                    image = gray,
                    board = board,
                    detectedCorners = corners,
                    detectedIds = ids,
                    rejectedCorners = rejectedImgPoints,
                    cameraMatrix = mtx,
                    distCoeffs = dist)

            corners_r, ids_r, rejectedImgPoints_r, recoveredIds_r = aruco.refineDetectedMarkers(
                    image = gray,
                    board = charuco_robot_r,
                    detectedCorners = corners_r,
                    detectedIds = ids_r,
                    rejectedCorners = rejectedImgPoints_r,
                    cameraMatrix = mtx,
                    distCoeffs = dist)
            corners_l, ids_l, rejectedImgPoints_l, recoveredIds_l = aruco.refineDetectedMarkers(
                    image = gray,
                    board = charuco_robot_l,
                    detectedCorners = corners_l,
                    detectedIds = ids_l,
                    rejectedCorners = rejectedImgPoints_l,
                    cameraMatrix = mtx,
                    distCoeffs = dist)

            watch_corners, watch_ids, watch_rejectedImgPoints, watch_recoveredIds = aruco.refineDetectedMarkers(
                    image = gray,
                    board = charuco_watch,
                    detectedCorners = watch_corners,
                    detectedIds = watch_ids,
                    rejectedCorners = watch_rejectedImgPoints,
                    cameraMatrix = mtx,
                    distCoeffs = dist)

            # Outline all of the markers detected in our image
            # left camera view
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners_r, borderColor=(0, 0, 255))
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners_l, borderColor=(0, 0, 255))
            QueryImg = aruco.drawDetectedMarkers(QueryImg, watch_corners, borderColor=(0, 0, 255))

            # Valid flags
            valid_board = False
            valid_robot_r = False
            valid_robot_l = False
            valid_watch = False


            # Detect the board, requires more than 3 markers on the board to be visible
            if ids is not None and len(ids) > 3:
                # Estimate the posture of the gridboard, seen from the left camera.
                pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist)
                if pose:
                    QueryImg = aruco.drawAxis(QueryImg, mtx, dist, rvec, tvec, board_marker_size)
                    board_to_cam_rot_mtx, board_origin = transform_from_board_to_camera(rvec, tvec, a_filter, board_origin)
                    valid_board = True

            # detect robot base markers
            if ids_r is not None and len(ids_r) >= 3:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners_r, robot_marker_size, mtx, dist)
                for ident, rvec, tvec in zip(ids_r, rvecs, tvecs):
                    if ident[0] == 0 or ident[0] == 1 or ident[0] == 3:
                        t = transform_from_base_to_camera(rvec, tvec)
                        base_marker1, base_marker2, base_marker3 = apply_filter(a_filter, ident[0], t, base_marker1, base_marker2, base_marker3)
                        QueryImg = aruco.drawAxis(QueryImg, mtx, dist, rvec, tvec, 0.02)
                x_axis, y_axis, z_axis, robot_origin = construct_new_axis_robot(base_marker2, base_marker1, base_marker3, False)
                # rot_mat is from robot to camera
                rot_mat = np.transpose(np.array([[x_axis[0], x_axis[1], x_axis[2]],[y_axis[0], y_axis[1], y_axis[2]], [z_axis[0], z_axis[1], z_axis[2]]]))
                board_origin_in_robot_frame = compute_board_origin_in_base_frame(board_origin, rot_mat, robot_origin)
                try:
                    Tb_r = compute_transformation_from_board_to_base(board_to_cam_rot_mtx, rot_mat, board_origin_in_robot_frame)
                    valid_robot_r = True
                except:
                    pass

            # detect left robot base markers
            if ids_l is not None and len(ids_l) >= 3:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners_l, robot_left_marker_size, mtx, dist)
                for ident, rvec, tvec in zip(ids_l, rvecs, tvecs):
                    #QueryImg = aruco.drawAxis(QueryImg, mtx, dist, rvec, tvec, 0.04)
                    if ident[0] == 0 or ident[0] == 2 or ident[0] == 3:
                        t = transform_from_base_to_camera(rvec, tvec)
                        base_marker_l1, base_marker_l2, base_marker_l3 = apply_filter(a_filter, ident[0], t, base_marker_l1, base_marker_l2, base_marker_l3)
                        QueryImg = aruco.drawAxis(QueryImg, mtx, dist, rvec, tvec, 0.02)
                x_axis, y_axis, z_axis, robot_origin_l = construct_new_axis_robot(base_marker_l2, base_marker_l1, base_marker_l3, True)
                # rot_mat_l is from robot to camera
                rot_mat_l = np.transpose(np.array([[x_axis[0], x_axis[1], x_axis[2]],[y_axis[0], y_axis[1], y_axis[2]], [z_axis[0], z_axis[1], z_axis[2]]]))
                board_origin_in_robot_frame = compute_board_origin_in_base_frame(board_origin, rot_mat_l, robot_origin_l)
                try:
                    Tb_l =  compute_transformation_from_board_to_base(board_to_cam_rot_mtx, rot_mat_l, board_origin_in_robot_frame)
                    valid_robot_l = True
                except:
                    pass

            # detect watch face markers
            if watch_ids is not None and len(watch_ids) >= 4:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(watch_corners, watch_marker_size, mtx, dist)
                for ident, rvec, tvec in zip(watch_ids, rvecs, tvecs):
                    if ident[0] == 0 or ident[0] == 1 or ident[0] == 3 or ident[0] == 2:
                        t = transform_from_base_to_camera(rvec, tvec)
                        watch_base_marker1, watch_base_marker2, watch_base_marker3, watch_base_marker4 = apply_filter_watch(a_filter, ident[0], t, watch_base_marker1, watch_base_marker2, watch_base_marker3, watch_base_marker4)
                        QueryImg = aruco.drawAxis(QueryImg, mtx, dist, rvec, tvec, 0.02)
                BA = watch_base_marker2 - watch_base_marker1
                DA = watch_base_marker3 - watch_base_marker1
                normal = np.cross(DA, BA)
                nnorm = np.linalg.norm(normal)
                if nnorm > 1e-6:
                    normal /= nnorm
                # compute the center point of all four markers
                watch_origin = compute_center_from_four_points(watch_base_marker2, watch_base_marker3, watch_base_marker1, watch_base_marker4)
                x_axis = watch_base_marker1 - watch_origin
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(normal, x_axis)
                rot_mat_w = np.transpose(np.array([[x_axis[0], x_axis[1], x_axis[2]],[y_axis[0], y_axis[1], y_axis[2]], [normal[0], normal[1], normal[2]]]))
                board_origin_in_watch_frame = compute_board_origin_in_base_frame(board_origin, rot_mat_w, watch_origin)
                try:
                    Tb_w = compute_transformation_from_board_to_base(board_to_cam_rot_mtx, rot_mat_w, board_origin_in_watch_frame)
                    valid_watch = True
                except:
                    pass

            self.pub_msg.header.stamp = rospy.Time.now()
            self.pub2_msg.header.stamp = rospy.Time.now()
            self.pub_watch_msg.header.stamp = rospy.Time.now()

            if valid_robot_r:
                Tb_r = np.linalg.inv(Tb_r)
                T = np.zeros((4,4))
                rad = 45.*np.pi/180.
                c = np.cos(rad)
                s = np.sin(rad)
                T[0,0] = c
                T[0,1] = -s
                T[1,0] = s
                T[1,1] = c
                T[2,2] = 1
                T[3,3] = 1
                # print(T)
                Tb_r = np.matmul(Tb_r, T)
            else:
                print(str(self.fr1) + ': Invalid right robot transformation. Using previous valid.')

            self.pub_msg.pose.position.x = Tb_r[0,3]
            self.pub_msg.pose.position.y = Tb_r[1,3]
            self.pub_msg.pose.position.z = Tb_r[2,3]
            T = copy.deepcopy(Tb_r)
            T[0:3,3] = np.array([0,0,0])
            quat_br = tf.transformations.quaternion_from_matrix(T)
            quat_br = quat_br/np.linalg.norm(quat_br)
            self.pub_msg.pose.orientation.x = quat_br[0]
            self.pub_msg.pose.orientation.y = quat_br[1]
            self.pub_msg.pose.orientation.z = quat_br[2]
            self.pub_msg.pose.orientation.w = quat_br[3]

            if valid_robot_l:
                Tb_l = np.linalg.inv(Tb_l)
                T = np.zeros((4,4))
                rad = 45.*np.pi/180.
                c = np.cos(rad)
                s = np.sin(rad)
                T[0,0] = c
                T[0,1] = -s
                T[1,0] = s
                T[1,1] = c
                T[2,2] = 1
                T[3,3] = 1
                # print(T)
                Tb_l = np.matmul(Tb_l, T)
            else:
                print(str(self.fr1) + ': Invalid left robot transformation. Using previous valid.')

            self.pub2_msg.pose.position.x = Tb_l[0,3]
            self.pub2_msg.pose.position.y = Tb_l[1,3]
            self.pub2_msg.pose.position.z = Tb_l[2,3]
            T = copy.deepcopy(Tb_l)
            T[0:3,3] = np.array([0,0,0])
            quat_bl = tf.transformations.quaternion_from_matrix(T)
            quat_bl = quat_bl/np.linalg.norm(quat_bl)
            self.pub2_msg.pose.orientation.x = quat_bl[0]
            self.pub2_msg.pose.orientation.y = quat_bl[1]
            self.pub2_msg.pose.orientation.z = quat_bl[2]
            self.pub2_msg.pose.orientation.w = quat_bl[3]

            if valid_watch:
                Tb_w = np.linalg.inv(Tb_w)
                # T = np.zeros((4,4))
                # rad = np.pi
                # c = np.cos(rad)
                # s = np.sin(rad)
                # T[0,0] = c
                # T[0,1] = -s
                # T[1,0] = s
                # T[1,1] = c
                # T[2,2] = 1
                # T[3,3] = 1
                # # print(T)
                # Tb_w = np.matmul(Tb_w, T)
            else:
                print(str(self.fr1) + ': Invalid watch face transformation. Using previous valid.')

            self.pub_watch_msg.pose.position.x = Tb_w[0,3]
            self.pub_watch_msg.pose.position.y = Tb_w[1,3]
            self.pub_watch_msg.pose.position.z = Tb_w[2,3]
            T = copy.deepcopy(Tb_w)
            T[0:3,3] = np.array([0,0,0])
            quat_bw = tf.transformations.quaternion_from_matrix(T)
            quat_bw = quat_bw/np.linalg.norm(quat_bw)
            self.pub_watch_msg.pose.orientation.x = quat_bw[0]
            self.pub_watch_msg.pose.orientation.y = quat_bw[1]
            self.pub_watch_msg.pose.orientation.z = quat_bw[2]
            self.pub_watch_msg.pose.orientation.w = quat_bw[3]

            self.pub.publish(self.pub_msg)
            self.pub2.publish(self.pub2_msg)
            self.pub_watch.publish(self.pub_watch_msg)

            if not valid_board:
                print(str(self.fr1) + ': Invalid board transformation. Using previous valid.')

            self.br.sendTransform(Tb_r[0:3,3], quat_br, rospy.Time.now(), 'robot', 'board')
            self.br.sendTransform(Tb_l[0:3,3], quat_bl, rospy.Time.now(), 'robot_left', 'board')
            self.br.sendTransform(Tb_w[0:3,3], quat_bw, rospy.Time.now(), 'watch_face', 'board')

            # # plot coordinate frame in 3D for debugging
            # try:
            #     QueryImg = draw_axis(QueryImg, rot_mat, robot_origin, mtx, dist)
            # except Exception as ex:
            #     print(ex)
            #     pass
            # try:
            #     QueryImg = draw_axis(QueryImg, rot_mat_l, robot_origin_l, mtx, dist)
            # except Exception as ex:
            #     print(ex)
            #     pass
            # try:
            #     QueryImg = draw_axis(QueryImg, rot_mat_w, watch_origin, mtx, dist)
            # except Exception as ex:
            #     print(ex)
            #     pass
            # # cv2.imwrite('./out/img.png', QueryImg)
            # cv2.imwrite('./out/imgA_{}.png'.format(self.fr1), QueryImg)
            # cv2.imwrite('./out/imgB_{}.png'.format(self.fr2), QueryImg1)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('cv_estimation', anonymous=True)

    track = Tracker()
    track.run()
