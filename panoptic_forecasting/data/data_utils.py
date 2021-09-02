# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import glob
import os
from collections import defaultdict
import json

import torch
from PIL import Image
import numpy as np
import cv2



def convert_bbox_ulbr_cwh(bboxes):
    orig_shape = bboxes.shape
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.reshape((-1, 4))
    else:
        bboxes = bboxes.reshape(-1, 4)
    cx = (bboxes[:, 0] + bboxes[:, 2])/2
    cy = (bboxes[:, 1] + bboxes[:, 3])/2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    if isinstance(bboxes, np.ndarray):
        return np.stack([cx, cy, w, h], axis=1).reshape(orig_shape)
    else:
        return torch.stack([cx, cy, w, h], dim=1).reshape(orig_shape)


def convert_bbox_cwh_ulbr(bboxes):
    orig_shape = bboxes.shape
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.reshape((-1, 4))
    else:
        bboxes = bboxes.reshape(-1, 4)
    x0 = bboxes[:, 0] - bboxes[:, 2] / 2
    x1 = bboxes[:, 0] + bboxes[:, 2] / 2
    y0 = bboxes[:, 1] - bboxes[:, 3] / 2
    y1 = bboxes[:, 1] + bboxes[:, 3] / 2

    if isinstance(bboxes, np.ndarray):
        return np.stack([x0, y0, x1, y1], axis=1).reshape(orig_shape)
    else:
        return torch.stack([x0, y0, x1, y1], dim=1).reshape(orig_shape)


def cityscapes_camera2intrinsics(camera):
    intrinsic = camera.get('intrinsic')
    fx = intrinsic.get('fx')
    fy = intrinsic.get('fy')
    u0 = intrinsic.get('u0')
    v0 = intrinsic.get('v0')

    assert isinstance(fx, float) and fx > 0.0, "weird 'fx' value"
    assert isinstance(fy, float) and fy > 0.0, "weird 'fy' value"
    assert isinstance(u0, float), "weird 'u0' value"
    assert isinstance(v0, float), "weird 'v0' value"


    intrinsics = np.zeros(4)
    intrinsics[0] = fx
    intrinsics[1] = fy
    intrinsics[2] = u0
    intrinsics[3] = v0

    return intrinsics


def cityscapes_camera2extrinsics(camera):
    vehicle_T_camera_flu = get_vehicle_T_camera_flu(camera)

    # We assume camera coordinates will be provided in RDF
    return np.matmul(vehicle_T_camera_flu, flu_T_rdf())


# Compose 4x4 affine transformation in the 3d space.
def compose_affine_3d(R=None, t=None):
    # Start with an identity transform.
    T = np.identity(4, dtype=np.float)

    # Assign rotation part if given.
    if R is not None:
        assert isinstance(R, np.ndarray) and R.shape == (3, 3)
        T[0:3, 0:3] = R

    # Assign translation part if given.
    if t is not None:
        assert isinstance(t, np.ndarray) and t.shape == (3, )
        T[0:3, 3] = t

    return T


# Transform converting from FLU (x-front; y-left; z-up) to RDF (x-right; y-down; z-front) coordinate at the same origin.
def rdf_T_flu():
    return compose_affine_3d(
        R=np.array([[0, -1,  0],
                    [0,  0, -1],
                    [1,  0,  0]], dtype=np.float),
        t=None)


# Transform converting from RDF (x-right; y-down; z-front) to FLU (x-front; y-left; z-up) coordinate at the same origin.
def flu_T_rdf():
    return compose_affine_3d(
        R=np.array([[ 0,  0, 1],
                    [-1,  0, 0],
                    [ 0, -1, 0]], dtype=np.float),
        t=None)


def get_vehicle_now_T_prev(speed, yaw_rate, delta_t):
    # The vehicle motion is stored in a 2d velocity model. The vehicle is assumed to move on a horizontal plan.
    # Thus, if the vehicle bumps or moves on a hill, this motion model can be inaccurate.

    # 2d pose representation:
    # The pose of the vehicle in a 2d top-down view coordinate is represented as [x, y, theta], where (x, y) specifies
    # the location, and theta specifies the heading direction. The coordinate is defined as z-up, and the vehicle moves
    # on the x-y plane.
    #
    # World coordinate:
    # In this function, define the global coordinate to be the vehicle's coordinate at the previous frame.
    # Thus, (x_prev = 0, y_prev = 0, theta_prev = 0).
    #
    # Steps:
    # 1. Calculate the current vehicle pose in the world coordinate.
    #    (Good reference for velocity model: Section 5.3 in https://ccc.inaoep.mx/~mdprl/documentos/CH5.pdf)
    # 2. Derive the 3x3 (3-dof) the relative poses between two frames in the world coordinate.
    # 3. Expand the 3x3 3-dof transform to a 4x4 6-dof transform assuming the vehicle travels on the x-y plane.

    # Handle special case where the angular velocity is close to 0. The vehicle is moving front.
    angle_rad_eps = 0.000175  # ~0.01 deg
    if abs(yaw_rate) < angle_rad_eps:
        x = delta_t * speed
        y = 0.0
        theta = 0.0
    else:
        # Follow Equation (5.9) in the reference by setting the previous pose to (0, 0, 0).
        r = speed / yaw_rate
        wt = yaw_rate * delta_t
        x = r * np.sin(wt)
        y = r - r * np.cos(wt)
        theta = wt

    # The rotation is along the z-axis.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    vehicle_prev_R_vehicle_now = np.array(
        [[cos_theta, -sin_theta, 0],
         [sin_theta, cos_theta,  0],
         [0,         0,          1]], dtype=np.float)

    # The translation is on the x-y plane (no changes in z).
    vehicle_prev_t_vehicle_now = np.array([x, y, 0])

    # Compose the 3D affine transform.
    T = compose_affine_3d(R=vehicle_prev_R_vehicle_now, t=vehicle_prev_t_vehicle_now)

    # T gives transformation from current frame to previous - we want inverse
    return np.linalg.inv(T), x, y, theta


# Obtain the 4x4 affine transform from FLU (x-front; y-left; z-up) camera's coordinate to vehicle's coordinate.
# NOTE: Most CV algorithms (including OpenCV) assume the camera's coordinate is RDF (x-right; y-down; z-front).
def get_vehicle_T_camera_flu(camera):
    assert isinstance(camera, dict)

    extrinsic = camera.get('extrinsic')
    assert isinstance(extrinsic, dict), "weird 'extrinsic'"

    # Read rotation.
    yaw = extrinsic.get('yaw')
    pitch = extrinsic.get('pitch')
    roll = extrinsic.get('roll')
    assert isinstance(roll, float) and isinstance(pitch, float) and isinstance(roll, float)

    # Follow Equation (3) to construct, rotation part (vehicle_R_camera) of the 4x4 transform, vehicle_T_camera
    s_y = np.sin(yaw)
    c_y = np.cos(yaw)
    s_p = np.sin(pitch)
    c_p = np.cos(pitch)
    s_r = np.sin(roll)
    c_r = np.cos(roll)
    vehicle_R_camera = np.array(
        [[c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
         [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
         [-s_p,      c_p * s_r,                   c_p * c_r]],
        dtype=np.float)

    # Follow Equation (3) to construct, translation part (vehicle_t_camera) of the 4x4 transform, vehicle_T_camera
    x = extrinsic.get('x')
    y = extrinsic.get('y')
    z = extrinsic.get('z')
    assert isinstance(x, float) and isinstance(y, float) and isinstance(z, float)
    vehicle_t_camera = np.array([x, y, z], dtype=np.float)

    # Compose the 3D affine transform.
    return compose_affine_3d(R=vehicle_R_camera, t=vehicle_t_camera)


# given array of [fx, fy, u0, v0], build 3x3 intrinsics matrix
def build_intrinsics_mat(intrinsics):
    result = np.eye(3)
    fx, fy, u0, v0 = intrinsics
    result[0,0] = fx
    result[1,1] = fy
    result[0,2] = u0
    result[1,2] = v0
    return result
