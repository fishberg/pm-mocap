#!/usr/bin/python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

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
    """
    D = np.array([[ np.cos(pitch),  np.sin(pitch),              0],
                  [-np.sin(pitch),  np.cos(pitch),              0],
                  [             0,              0,              1]])

    C = np.array([[             1,              0,              0],
                  [             0,   np.cos(roll),   np.sin(roll)],
                  [             0,  -np.sin(roll),   np.cos(roll)]])

    B = np.array([[   np.cos(yaw),    np.sin(yaw),              0],
                  [  -np.sin(yaw),    np.cos(yaw),              0],
                  [             0,              0,              1]])
    """

    #A = np.matmul(np.matmul(B,C),D)
    A = np.matmul(np.matmul(D,C),B)
    return A

def deg2rad(deg):
    return np.pi/180 * deg

def rad2deg(rad):
    return 180/np.pi * rad

class Camera:
    def __init__(self,x,y,z,roll,pitch,yaw,wfov,hfov):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.wfov = wfov
        self.hfov = hfov

    def mk_fov(self):
        rot = rot_mat(self.roll, self.pitch, self.yaw)

        line_step = np.linspace([0,0,0],[5,0,0],101).T
        print(line_step.shape)
        line_center = np.matmul(rot, line_step)
        line_tl = np.matmul(rot, np.matmul(rot_mat(0, self.hfov/2,-self.wfov/2), line_step))
        line_tr = np.matmul(rot, np.matmul(rot_mat(0, self.hfov/2, self.wfov/2), line_step))
        line_bl = np.matmul(rot, np.matmul(rot_mat(0,-self.hfov/2,-self.wfov/2), line_step))
        line_br = np.matmul(rot, np.matmul(rot_mat(0,-self.hfov/2, self.wfov/2), line_step))

        loc = np.array([self.x, self.y, self.z])

        rec = np.array([line_tl[:,20], line_tr[:,20], line_br[:,20], line_bl[:,20], line_tl[:,20]]).T
        print(rec)
        print(rec.shape)
        print(line_tl)
        print(line_tl.shape)

        """
        line_center = (line_center.T + loc).T
        line_wsub = (line_wsub.T + loc).T
        line_wadd = (line_wadd.T + loc).T
        line_hsub = (line_hsub.T + loc).T
        line_hadd = (line_hadd.T + loc).T
        """
        """
        assert all(line_hsub[0,:] == line_hadd[0,:])
        assert all(line_hsub[1,:] == line_hadd[1,:])
        assert all(-line_hsub[2,:] == line_hadd[2,:])

        assert all(line_wsub[0,:] == line_wadd[0,:])
        assert all(-line_wsub[1,:] == line_wadd[1,:])
        assert all(line_wsub[2,:] == line_wadd[2,:])
        """

        return line_center, line_tl, line_tr, line_bl, line_br, rec
    
    def plot(self, ax):
        i = 0
        colors = ['green','red','orange','blue','purple','black']
        for line in self.mk_fov():
            xx,yy,zz = line
            ax.plot(xx,yy,zz,c=colors[i])
            i+=1
            #break

class Webcam(Camera):
    def __init__(self,x,y,z,roll,pitch,yaw):
        super(Webcam,self).__init__(x,y,z,roll,pitch,yaw,deg2rad(60),deg2rad(60*9/16))

jj = 1
fig = plt.figure()
def mk_plot():
    global fig, jj
    ax = fig.add_subplot(330 + jj,projection='3d')
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

#fig,ax = mk_plot()
#
#xs = [0]
#ys = [0]
#zs = [0]
#
#ax.scatter(xs,ys,zs, c='r', marker='o')

#cam1 = Webcam(-2,-2,0,0,deg2rad(30),deg2rad(-45))
#cam1 = Webcam(0,0,0,deg2rad(90),deg2rad(0),deg2rad(0))
#cam2 = Webcam(2,-2,0,0,deg2rad(30),deg2rad(-135))
#cam3 = Webcam(2,2,0,0,deg2rad(30),deg2rad(135))
#cam1.plot(ax)
#cam2.plot(ax)
#cam3.plot(ax)



#plt.show()

curr = 0
for curr in range(0,361,45):
    fig,ax = mk_plot()
    cam1 = Webcam(0,0,0,deg2rad(0),deg2rad(0),deg2rad(curr))
    cam1.plot(ax)
    ax.set_title(str(curr))

plt.show()
