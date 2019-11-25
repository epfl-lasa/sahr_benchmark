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
# import packages
import numpy as np
import cv2
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

# define function to draw axis
def draw_axis(img, rot_mat, t, K, dist):
    '''
        params:
            img: QueryImg, camera image to draw the axis
            rot_mat: rotation matrix that transform a point from the frame in question to camera frame
            t: translation vector (base origin in camera's frame)
            k: camera intrinsic param, camera matrix
            dist: camera intrinsic param, distortion_coefficients0
    '''
    rotV = cv2.Rodrigues(rot_mat)[0]
    points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, dist)
    
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 3)
    return img

def transform_from_board_to_camera(rvec, tvec, filter, past_location):
    '''
        params: 
            rvec: rotation vector, from board frame to camera frame
            tvec: translation vector
            filter: float, loc = filter * past_loc + (1 - filter) * new_loc
            past_location: past location of the board, needed for filtering

        return: a numpy array that represents the board's location in camera's frame, 
        assuming rvec and tvec are computed using camera's calibration data.
    '''
    R, _ = cv2.Rodrigues(rvec)
    R = np.array(R)
    tr = np.transpose(np.array(tvec))
    zero = np.zeros((4,1))
    zero[3] = 1.
    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[0:3,3] = tr.reshape((3,))
    T[3,3] = 1.
    t = np.matmul(T, zero)
    t[0][0] = filter * past_location[0] + (1-filter)*t[0][0]
    t[1][0] = filter * past_location[1] + (1-filter)*t[1][0]
    t[2][0] = filter * past_location[2] + (1-filter)*t[2][0]
    return R, np.array([t[0][0], t[1][0], t[2][0]])

def transform_from_base_to_camera(rvec, tvec):
    '''
        params:
            rvec: rotation vector for a single marker
            tvecs: translation vector for a single marker
        return:
            t: numpy array, location of origin (in a single marker's frame) in camera's frame
             calculated from given information 
    '''
    R, _ = cv2.Rodrigues(rvec)
    R = np.array(R)
    tr = np.transpose(np.array(tvec))
    zero = np.zeros((4,1))
    zero[3] = 1.
    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[0:3,3] = tr.reshape((3,))
    T[3,3] = 1.
    t = np.matmul(T, zero)
    return t

def compute_center_from_three_points(u_diag, l_diag, cor):
    '''
        find the center of the robot frame based on three corners, since
        camera can only see three corners for most time
        
        params:
            u_diag: upper diagonal point (the marker further away from the camera (id 1 or id 2), coordinate expressed in camera's frame)
            l_diag: lower diagonal point (different based on robot)
            cor: the marker which is different from above two.

        return: 
            the coordinate of center of robot frame (in camera's frame)
    '''
    diag = u_diag - l_diag
    diag_length = np.linalg.norm(diag)
    diag_norm = diag / diag_length
    point_avg = l_diag + diag_norm * diag_length / 2.

    side_1 = u_diag - cor
    side_2 = l_diag - cor
    inv_diag = side_1 + side_2
    inv_diag_len = np.linalg.norm(inv_diag)
    inv_diag_norm = inv_diag / inv_diag_len

    point_avg = (point_avg + cor + inv_diag_norm * inv_diag_len / 2.) / 2.
    return point_avg

def compute_board_origin_in_base_frame(board_origin, rot_mat, base_center):
    """
        compute the coordinate of board origin in robot frame:

        param:
            board_origin: board's origin coordinate in camera's frame
            rot_mat: rotation matrix from base's frame to camera
            base_center: origin of robot/watch frame, computed using compute_center_from_three_point or compute_center_from_four_point
        return:
            board's coornidate in robot/watch frame.
    """
    T = np.zeros((4,4))
    T[0:3,0:3] = np.transpose(rot_mat)
    tr = -np.matmul(np.transpose(rot_mat),np.transpose(base_center))
    T[0:3,3] = tr.reshape((3,))
    T[3,3] = 1.
    # print(T)
    tt = np.zeros((4,1))
    tt[0:3] = np.array([board_origin]).reshape((3,1))
    tt[3] = 1.
    return np.matmul(T,tt)

def compute_center_from_four_points(ul, ur, ll, lr):
    point_avg = (ul + ur + ll + lr) / 4.
    return point_avg

def apply_filter(a, id, t, base_marker1, base_marker2, base_marker3):
    '''
        apply filtering to robot frames specifically.
        a: amount of filtering
        id: ChArUco Marker Id
        t: current computed location info
        base_marker#: previous stored location info about each marker.
    '''
    if id == 0:
        t[0][0] = a*base_marker1[0] + (1-a)*t[0][0]
        t[1][0] = a*base_marker1[1] + (1-a)*t[1][0]
        t[2][0] = a*base_marker1[2] + (1-a)*t[2][0]
        base_marker1 = np.array([t[0][0], t[1][0], t[2][0]])
    elif id == 1 or id == 2:
        t[0][0] = a*base_marker2[0] + (1-a)*t[0][0]
        t[1][0] = a*base_marker2[1] + (1-a)*t[1][0]
        t[2][0] = a*base_marker2[2] + (1-a)*t[2][0]
        base_marker2 = np.array([t[0][0], t[1][0], t[2][0]])
    elif id == 3:
        t[0][0] = a*base_marker3[0] + (1-a)*t[0][0]
        t[1][0] = a*base_marker3[1] + (1-a)*t[1][0]
        t[2][0] = a*base_marker3[2] + (1-a)*t[2][0]
        base_marker3 = np.array([t[0][0], t[1][0], t[2][0]])
    return base_marker1, base_marker2, base_marker3

def apply_filter_watch (a, id, t, watch_base_marker1, watch_base_marker2, watch_base_marker3, watch_base_marker4):
    '''
        Like apply_filter, but for watch base only.
    '''
    if id == 0:
        t[0][0] = a*watch_base_marker1[0] + (1-a)*t[0][0]
        t[1][0] = a*watch_base_marker1[1] + (1-a)*t[1][0]
        t[2][0] = a*watch_base_marker1[2] + (1-a)*t[2][0]
        watch_base_marker1 = np.array([t[0][0], t[1][0], t[2][0]])
    elif id == 1:
        t[0][0] = a*watch_base_marker2[0] + (1-a)*t[0][0]
        t[1][0] = a*watch_base_marker2[1] + (1-a)*t[1][0]
        t[2][0] = a*watch_base_marker2[2] + (1-a)*t[2][0]
        watch_base_marker2 = np.array([t[0][0], t[1][0], t[2][0]])
    elif id == 3:
        t[0][0] = a*watch_base_marker3[0] + (1-a)*t[0][0]
        t[1][0] = a*watch_base_marker3[1] + (1-a)*t[1][0]
        t[2][0] = a*watch_base_marker3[2] + (1-a)*t[2][0]
        watch_base_marker3 = np.array([t[0][0], t[1][0], t[2][0]])
    elif id == 2:
        t[0][0] = a*watch_base_marker4[0] + (1-a)*t[0][0]
        t[1][0] = a*watch_base_marker4[1] + (1-a)*t[1][0]
        t[2][0] = a*watch_base_marker4[2] + (1-a)*t[2][0]
        watch_base_marker4 = np.array([t[0][0], t[1][0], t[2][0]])
    return watch_base_marker1, watch_base_marker2, watch_base_marker3, watch_base_marker4

def compute_transformation_from_board_to_base(board_to_cam_rot_mtx, rot_mat, board_origin_in_base_frame):
    '''
        Compute transformation matrix from board frame to base frame.
        param:
            board_to_cam_rot_mat: rotation matrix from board to camera
            rot_mat: rotation matrix from base (robot, watch) to camera
            board_origin_in_base_frame: board's origin coordinate measured in base frame.
        return: 
            translation matrix, translates a point in board frame to base frame.
    '''
    robot_to_board_mtx = np.matmul(np.transpose(board_to_cam_rot_mtx), rot_mat)
    T = np.zeros((4,4))
    T[0:3,0:3] = robot_to_board_mtx.T
    tr = board_origin_in_base_frame[:3]
    T[0:3,3] = tr.reshape((3,))
    T[3,3] = 1.
    Tboard_base = copy.deepcopy(T)
    return Tboard_base

def construct_new_axis_robot(base_marker_upper, base_marker_ll, base_marker_lr, left_robot):
    '''
        Construct new robot axis.
        param: base_marker_upper: location array of the marker that is further away from the camera
                                    (upper ChArUco marker's location array, either with id 1 or 2 in our setup, depends on the robot)
               base_marker_ll: base_marker lower left, location array of the marker with id 0 in our setup.
               base_marker_lr: base_marker lower right, location array of the marker with id 3 in our setup.
               left_robot: boolean, indicate which robot it is
        return:
               unit length axis in camera's frame that represent the robot base frame.
    '''
    BA = base_marker_upper - base_marker_ll
    DA = base_marker_lr - base_marker_ll
    normal = np.cross(DA, BA)
    nnorm = np.linalg.norm(normal)
    if nnorm > 1e-6:
        normal /= nnorm
    # compute the center point of all four markers, the origin of robot coordinate frames
    if left_robot:
        origin = compute_center_from_three_points(base_marker_upper, base_marker_ll, base_marker_lr)
    else:
        origin = compute_center_from_three_points(base_marker_upper, base_marker_lr, base_marker_ll)

    x_axis = base_marker_ll - origin
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    return x_axis, y_axis, normal, origin