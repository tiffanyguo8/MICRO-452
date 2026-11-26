import numpy as np

L = 93

m = np.zeros((3, 3))
M = np.load('M.npy')
R = np.diag([0.01, 0.01, 1e-5])

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def f(x, y, theta, v_r, v_l, dt, L=L):
    f = np.array([x + (v_r + v_l) * np.cos(theta) * dt / 2, 
         y + (v_r + v_l) * np.sin(theta) * dt / 2, 
         wrap_angle(theta + (v_r - v_l) * dt / L)])
    return f

def kalman(x_old, v_r, v_l, P_old, dt, M=M, R=R, x_measured=None):
    # Units : 
    # x, y, L in mm; theta in radian (-π, π)
    # v_r, v_l in mm/s
    # dt in s
    # M in (mm/s)^2
    # R and P in diag(mm^2, mm^2, rad^2)
    # All arrays should be NumPy arrays of floats
    x, y, theta = x_old[0], x_old[1], x_old[2]
    G = np.array([[np.cos(theta) * dt / 2, np.cos(theta) * dt / 2], 
                  [np.sin(theta) * dt / 2, np.sin(theta) * dt / 2], 
                  [dt / L, -dt / L]])
    F = np.array([[1, 0, -(v_r + v_l) * np.sin(theta) * dt / 2], 
         [0, 1, (v_r + v_l) * np.cos(theta) * dt / 2], 
         [0, 0, 1]])
    x_pred = f(x, y, theta, v_r, v_l, dt)
    P_pred = F @ P_old @ F.T + G @ M @ G.T
    if x_measured is not None:
        i = x_measured - x_pred
        i[2] = wrap_angle(i[2])
        S = P_pred + R
        K = P_pred @ np.linalg.inv(S)
        x_est = x_pred + K @ i
        x_est[2] = wrap_angle(x_est[2])
        P_est = (np.identity(3) - K) @ P_pred
        return x_est, P_est
    else : 
        return x_pred, P_pred