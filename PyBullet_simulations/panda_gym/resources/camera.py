'''---------------------------------'''
'''
Function that loads the different cameras on the pybullet simulation environment
    # 430/435i fov = 91.2 nearVal = 0.3  farVal=10   480x640
    # camera1  fov = 80   nearVal=0.02   farVal=300  800x800
    # camera2  fov = 80   nearVal=0.02   farVal=300  800x800
'''

import numpy as np
import matplotlib.pyplot as plt
import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import pybullet as p


def camera(client_id, pixelWidth,pixelHeight,focal_length, farVal, nearVal, fov, cameraEyePosition,save_id, yaw = 0, pitch = 0, roll = 0, renderer = p.ER_TINY_RENDERER, RGB_show = False, RGB_save = False, DEPTH_show = False, DEPTH_save = False):

    cameraTargetPosition = [cameraEyePosition[0]+focal_length*np.cos(np.pi * np.abs(pitch)/180), cameraEyePosition[1], cameraEyePosition[2]-focal_length*np.sin(np.pi * np.abs(pitch)/180)]
    # p.addUserDebugLine([0,0,0],cameraEyePosition)
    # p.addUserDebugLine([0,0,0],cameraTargetPosition)

    ''' Extrinsic matrix '''
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = cameraTargetPosition, distance = focal_length, yaw = yaw, pitch = pitch, roll = roll, upAxisIndex = 2)
    ''' Projection matrix '''
    projectionMatrix = p.computeProjectionMatrixFOV(fov = fov,aspect = pixelWidth / pixelHeight, nearVal = nearVal, farVal = farVal)
    _,_,RGB,Depth, Segmentation = p.getCameraImage(
                                pixelWidth,
                                pixelHeight,
                                viewMatrix = viewMatrix,
                                projectionMatrix = projectionMatrix,
                                shadow = False,
                                lightDirection = [1,1,1],
                                lightDistance= 0.0,
                                lightAmbientCoeff = np.random.uniform(0.4,0.75),
                                lightDiffuseCoeff= np.random.uniform(0.4,0.75),
                                renderer = renderer,
                                physicsClientId = client_id)


    ''' Extraction of the RGB pixels as array '''
    rgb_array = np.array(RGB)
    rgb_array = rgb_array[:, :, :3]

    ''' Extraction of the Depth pixels as array '''
    depth_array = np.array(Depth)

    ''' IF you want to extract the real depth values in world coordinates'''
    #depth_array = farVal * nearVal / (farVal - (farVal - nearVal) * depth_array)

    ''' Show the RGB image with matplotlib '''
    if RGB_show:
        plt.figure()
        image = plt.imshow(rgb_array, interpolation='none', animated=True, label="RGB")
        ax = plt.gca()
        ax.plot([0])
        plt.pause(0.01)

    ''' Save the  RGB image with matplotlib '''
    if RGB_save:
        fig = plt.figure()
        plt.imshow(rgb_array, interpolation='none', animated=True, label="RGB")
        plt.axis('off')
        plt.gca().set_position((0, 0, 1, 1))
        DPI = fig.get_dpi()
        fig.set_size_inches(pixelWidth/float(DPI),pixelHeight/float(DPI))
        plt.savefig(currentdir+'/Data_collected/RGB_home/'+ save_id +'.png',dpi='figure')
        print('Home position RGB image saved in:     '+ currentdir +'/RGB_home/home_img.png')

    ''' Show the DEPTH image with matplotlib '''
    if DEPTH_show:
        plt.figure()
        image = plt.imshow(depth_array, cmap='gray', vmin=0, vmax=1,label="DEPTH")
        ax = plt.gca()
        ax.plot([0])
        plt.pause(0.01)

    ''' Save the DEPTH image with matplotlib '''
    if DEPTH_save:
        fig = plt.figure()
        plt.imshow(depth_array, cmap='gray', vmin=0, vmax=1,label="DEPTH")
        plt.axis('off')
        plt.gca().set_position((0, 0, 1, 1))
        DPI = fig.get_dpi()
        fig.set_size_inches(pixelWidth/float(DPI),pixelHeight/float(DPI))
        plt.savefig(currentdir+'/Data_collected/D_home_img/'+ save_id +'.png',dpi='figure')
        print('Home position DEPTH image saved in:     '+ currentdir +'/D_home_img/'+ save_id +'.png')

    return rgb_array, depth_array
