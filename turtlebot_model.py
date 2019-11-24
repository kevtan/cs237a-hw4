import numpy as np
import math

EPSILON_OMEGA = 1e-3

def compute_dynamics(X, U, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
        X: np.array[3,] - Turtlebot state (x, y, theta).
        U: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
        g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    (x, y, theta), (v, omega) = X, U
    if abs(omega) < EPSILON_OMEGA:
        costh = np.cos(theta)
        sinth = np.sin(theta)
        g = np.array([
            x + v * costh * dt,
            y + v * sinth * dt,
            theta
        ])
        Gx = np.array([
            [1, 0, -v * sinth * dt],
            [0, 1, v * costh * dt],
            [0, 0, 1]
        ])
        Gu = np.array([
            [costh * dt, v * ((sinth * dt**2) / 2 - sinth * dt**2)],
            [sinth * dt, v * ((-costh * dt**2)/ 2 + costh * dt**2)],
            [0, dt]
        ])
    else:
        theta_new = theta + omega * dt
        costh = np.cos(theta)
        sinth = np.sin(theta)
        costhn = np.cos(theta_new)
        sinthn = np.sin(theta_new)
        g = np.array([
            x + (v/omega) * (sinthn - sinth),
            y + (v/omega) * (costh - costhn),
            theta_new
        ])
        Gx = np.array([
            [1, 0, (v/omega) * (costhn - costh)],
            [0, 1, (v/omega) * (sinthn - sinth)],
            [0, 0, 1]
        ])
        Gu = np.array([
            [(sinthn - sinth) / omega, v * (-sinthn/(omega**2) + (costhn * dt)/omega + sinth/(omega**2))],
            [(costh - costhn) / omega, v * (costhn/(omega**2) + (sinthn * dt)/omega - costh/(omega**2))],
            [0, dt]
        ])
    if not compute_jacobians:
        return g
    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    # extract data from arguments
    alpha_w, r_w = line
    x_bw, y_bw, theta_bw = x
    x_cb, y_cb, theta_cb = tf_base_to_camera
    # find alpha in the camera frame
    alpha_c = alpha_w - theta_bw - theta_cb
    # find r in the camera frame
    sinth, costh = np.sin(theta_bw), np.cos(theta_bw)
    rotation = np.array([
        [costh, -sinth],
        [sinth, costh]
    ])
    c_offset = np.matmul(rotation, [x_cb, y_cb])
    c_x, c_y = [x_bw, y_bw] + c_offset
    # compute the min distance frame line to camera origin
    a, b, c = -np.cos(alpha_w), -np.sin(alpha_w), r_w
    r_c = a * c_x + b * c_y + c
    h = np.array([alpha_c, r_c])
    if compute_jacobian:
        dr_dxbw = a
        dr_dybw = b
        dcx_dthetabw = -sinth * x_cb - costh * y_cb
        dcy_dthetabw = costh * x_cb - sinth * y_cb
        dr_dtheta_bw = a * dcx_dthetabw + b * dcy_dthetabw
        Hx = np.array([
            [0, 0, -1],
            [dr_dxbw, dr_dybw, dr_dtheta_bw]
        ])
        return h, Hx
    return h


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
