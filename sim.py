#!/usr/bin/python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy

import time

def rot_mat(roll, pitch, yaw):
    roll = -roll
    pitch = -pitch
    yaw = -yaw
    D = np.array([[   np.cos(yaw),    np.sin(yaw),              0],
                  [  -np.sin(yaw),    np.cos(yaw),              0],
                  [             0,              0,              1]])

    C = np.array([[ np.cos(pitch),              0, -np.sin(pitch)],
                  [             0,              1,              0],
                  [ np.sin(pitch),              0,  np.cos(pitch)]])

    B = np.array([[             1,              0,              0],
                  [             0,   np.cos(roll),   np.sin(roll)],
                  [             0,  -np.sin(roll),   np.cos(roll)]])

    A = D @ C @ B
    return A

def deg2rad(deg):
    return np.pi/180 * deg

def rad2deg(rad):
    return 180/np.pi * rad

class Camera:
    def __init__(self,x,y,z,roll,pitch,yaw,wfov,hfov,wpix,hpix):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.wfov = wfov
        self.hfov = hfov
        self.wpix = wpix
        self.hpix = hpix

    def mk_fov(self):
        rot = rot_mat(self.roll, self.pitch, self.yaw)

        line_step = np.linspace([0,0,0],[5,0,0],101).T

        line_center = rot @ line_step

        line_tl = rot @ rot_mat(0, self.hfov/2,-self.wfov/2) @ line_step
        line_tr = rot @ rot_mat(0, self.hfov/2, self.wfov/2) @ line_step
        line_bl = rot @ rot_mat(0,-self.hfov/2,-self.wfov/2) @ line_step
        line_br = rot @ rot_mat(0,-self.hfov/2, self.wfov/2) @ line_step

        loc = np.array([self.x, self.y, self.z])

        rec = np.array([ line_tl[:,20],
                         line_tr[:,20],
                         line_br[:,20],
                         line_bl[:,20],
                         line_tl[:,20]]).T


        line_center = (line_center.T + loc).T
        line_tl = (line_tl.T + loc).T
        line_tr = (line_tr.T + loc).T
        line_bl = (line_bl.T + loc).T
        line_br = (line_br.T + loc).T
        rec = (rec.T + loc).T

        return line_center, line_tl, line_tr, line_bl, line_br, rec
    
    def plot(self, ax):
        i = 0
        colors = ['green','red','orange','blue','purple','black','gray','gray','gray','gray']
        for line in self.mk_fov():
            xx,yy,zz = line
            ax.plot(xx,yy,zz,c=colors[i])
            i+=1
            #break

    def visable(self,x,y,z):
        point = np.array([x,y,z]).T

        rot = rot_mat(self.roll, self.pitch, self.yaw)
        line_step = np.array([1,0,0]).T

        line_center = rot @ line_step

        line_tl = rot @ rot_mat(0, self.hfov/2,-self.wfov/2) @ line_step
        line_tr = rot @ rot_mat(0, self.hfov/2, self.wfov/2) @ line_step
        line_bl = rot @ rot_mat(0,-self.hfov/2,-self.wfov/2) @ line_step
        line_br = rot @ rot_mat(0,-self.hfov/2, self.wfov/2) @ line_step

        v1 = line_tr - line_tl 
        v2 = line_bl - line_tl

        A = np.array([v1,v2]).T

        P = A @ np.linalg.inv(A.T @ A) @ A.T

        proj = P @ point
        dist = np.linalg.norm(point - proj)

        rotInv = rot_mat(-self.roll, -self.pitch, -self.yaw)
        img = rotInv @ proj
        flat = img[1:]

        # create unit corners
        corner_tl = rot_mat(0, self.hfov/2,-self.wfov/2) @ line_step
        corner_tr = rot_mat(0, self.hfov/2, self.wfov/2) @ line_step
        corner_bl = rot_mat(0,-self.hfov/2,-self.wfov/2) @ line_step
        corner_br = rot_mat(0,-self.hfov/2, self.wfov/2) @ line_step

        # adjust to the right distance
        corner_tl /= corner_tl[0]
        corner_tr /= corner_tr[0]
        corner_bl /= corner_bl[0]
        corner_br /= corner_br[0]

        # remove flattened dimension
        corner_tl = corner_tl[1:]
        corner_tr = corner_tr[1:]
        corner_bl = corner_bl[1:]
        corner_br = corner_br[1:]

        in_front = np.dot(point,line_center) >= 0
        visable = in_front and \
                  corner_tl[0] <= flat[0] <= corner_tr[0] and \
                  corner_tl[1] <= flat[1] <= corner_bl[1]
        pixel = None

        if visable:
            xi = flat[0] - corner_tl[0]
            yi = flat[1] - corner_tl[1]
            xstep = (corner_tr[0] - corner_tl[0]) / self.wpix
            ystep = (corner_bl[1] - corner_tl[1]) / self.hpix
            pixel = (xi / xstep, yi / ystep)

        return visable, pixel

    def ray(self,x,y):
        rot = rot_mat(self.roll, self.pitch, self.yaw)
        line_step = np.array([1,0,0]).T

        # create unit corners
        corner_tl = rot_mat(0, self.hfov/2,-self.wfov/2) @ line_step
        corner_tr = rot_mat(0, self.hfov/2, self.wfov/2) @ line_step
        corner_bl = rot_mat(0,-self.hfov/2,-self.wfov/2) @ line_step
        corner_br = rot_mat(0,-self.hfov/2, self.wfov/2) @ line_step

        # adjust to the right distance
        corner_tl /= corner_tl[0]
        corner_tr /= corner_tr[0]
        corner_bl /= corner_bl[0]
        corner_br /= corner_br[0]

        # remove flattened dimension
        corner_tl = corner_tl[1:]
        corner_tr = corner_tr[1:]
        corner_bl = corner_bl[1:]
        corner_br = corner_br[1:]

        xstep = (corner_tr[0] - corner_tl[0]) / self.wpix
        ystep = (corner_bl[1] - corner_tl[1]) / self.hpix
        p = (x * xstep + corner_tl[0], y * ystep + corner_tl[1])

        relpoint = np.array([np.sqrt(1-p[0]**2-p[1]**2), p[0], p[1]])
        point = rot @ relpoint

        return np.linspace([0,0,0],5*point,101).T

class Webcam(Camera):
    def __init__(self,x,y,z,roll,pitch,yaw):
        super(Webcam,self).__init__(x,y,z,roll,pitch,yaw,deg2rad(60),deg2rad(44.048625674),1280,960)

class Track:
    def __init__(x,y,z):
        self.x = x
        self.y = y
        self.z = z

jj = 1
fig = plt.figure()
def mk_plot3D():
    global fig, jj
    ax = fig.add_subplot(120 + jj,projection='3d')
    jj += 1
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim([-2,2])
    ax.set_ylim([2,-2])
    ax.set_zlim([2,-2])

    grounded = np.linspace([0,0,0],[5,0,0],50).T
    xx,yy,zz = grounded
    ax.plot(xx,yy,zz,c='black')

    return fig, ax

def mk_plot2D():
    global fig, jj
    ax = fig.add_subplot(120 + jj)
    jj += 1
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    ax.set_xlim([0,1280])
    ax.set_ylim([960,0])

    return fig, ax

fig,ax = mk_plot3D()
cam1 = Webcam(0,0,0,deg2rad(0),deg2rad(45),deg2rad(90))
cam1.plot(ax)
fig,ax2 = mk_plot2D()

tracks = [(1,0,0),(1,.1,0), (-1,0,0)]
for track in tracks:
    xx,yy,zz = track
    ax.scatter(xx,yy,zz,c='brown')
    visable, pixel = cam1.visable(*track)
    if visable:
        xx,yy = pixel
        ax2.scatter(xx,yy, c='brown')

pixels = [(1280//2,960//4)]
for pixel in pixels:
    line = cam1.ray(*pixel)
    xx,yy,zz = line
    ax.plot(xx,yy,zz,c='cyan')
    ax2.scatter(pixel[0],pixel[1],c='cyan')


plt.show()
