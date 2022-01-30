import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

from tools.poisson_disk_sample import generate_points


aux_video_path = "input_gdisko_gauss_r10_s10"  # path to the result gauss r10 s15 sequence
keyframes_path = 'keyframes'

flow_fwd_path = "flow_fwd"
flow_bwd_path = "flow_bwd"


def generate_flow_and_keyframes(method, video_path, output_path, params=[], backward=False, last_index=109, to_gray=False):
    frame_index = 0
    if backward:
        frame_index = last_index
    # Read the video and first frame
    old_frame = cv2.imread(video_path + f'/{frame_index:03}.png')

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    samples = generate_points(old_frame.shape[1], old_frame.shape[0])
    path = keyframes_path + f'/{frame_index:03}.png'
    cv2.imwrite(path, np.ones_like(old_frame) * 255)

    while True:
        if backward:
            frame_index -= 1
            if frame_index < 0:
                break
        else:
            frame_index += 1
        # Read the next frame
        new_frame = cv2.imread(video_path + f'/{frame_index:03}.png')
        frame_copy = new_frame
        if new_frame is None:
            break

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        with open(f'{output_path}/{frame_index:03}.A2V2f', "wb") as of:
            import struct
            of.write(struct.pack('iii', flow.shape[0], flow.shape[1], 8))
            for y in range(flow.shape[0]):
                for x in range(flow.shape[1]):
                    of.write(struct.pack('ff', -flow[y][x][0], -flow[y][x][1]))

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        for i in range(len(samples)):
            x = samples[i][0]
            y = samples[i][1]
            if x >= flow.shape[1] or x < 0 or y >= flow.shape[0] or y < 0:
                continue
            f = flow[int(y)][int(x)]
            samples[i] = samples[i][0] + f[0], samples[i][1] + f[1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x=samples[:, 0], y=samples[:, 1] * -1 + flow.shape[0], c='r', s=20)
        ax.set_aspect(1)
        ax.set_xlim([0, flow.shape[1]])
        ax.set_ylim([0, flow.shape[0]])
        ax.axis('off')
        fig.savefig('temp.png')
        plt.close(fig)
        temp = cv2.imread('temp.png')
        cv2.imshow("dots", temp)

        empty_patches = 0
        for i in range(flow.shape[0] // 32):
            for j in range(flow.shape[1] // 32):
                contains_point = False
                for s in samples:
                    if s[0] // 32 == j and s[1] // 32 == i:
                        contains_point = True
                        break
                if not contains_point:
                    empty_patches += 1
        if empty_patches > 120:
            samples = np.concatenate((samples, generate_points(old_frame.shape[1], old_frame.shape[0])))
            path = keyframes_path + f'/{frame_index:03}.png'
            cv2.imwrite(path, np.ones_like(old_frame)*255)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

        # Update the previous frame
        old_frame = new_frame
    os.remove('temp.png')


def generate_flow(method, video_path, output_path, params=[], backward=False, last_index=109, to_gray=False):
    frame_index = 0
    if backward:
        frame_index = last_index
    # Read the video and first frame
    old_frame = cv2.imread(video_path + f'/{frame_index:03}.png')

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        if backward:
            frame_index -= 1
            if frame_index < 0:
                break
        else:
            frame_index += 1
        # Read the next frame
        new_frame = cv2.imread(video_path + f'/{frame_index:03}.png')
        frame_copy = new_frame
        if new_frame is None:
            break

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        with open(f'{output_path}/{frame_index:03}.A2V2f', "wb") as of:
            import struct
            of.write(struct.pack('iii', flow.shape[0], flow.shape[1], 8))
            for y in range(flow.shape[0]):
                for x in range(flow.shape[1]):
                    of.write(struct.pack('ff', -flow[y][x][0], -flow[y][x][1]))

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

        # Update the previous frame
        old_frame = new_frame


def preprocess_video(input_directory):
    ls = os.listdir(input_directory)
    generate_flow_and_keyframes(cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM).calc, input_directory,
                                flow_fwd_path, [], to_gray=True)
    generate_flow(cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM).calc, input_directory,
                  flow_bwd_path, [], backward=True, last_index=len(ls) - 1, to_gray=True)


if __name__ == '__main__':
    if not os.path.exists(flow_fwd_path):
        os.mkdir(flow_fwd_path)
    if not os.path.exists(flow_bwd_path):
        os.mkdir(flow_bwd_path)
    if not os.path.exists(keyframes_path):
        os.mkdir(keyframes_path)
    preprocess_video(sys.argv[1])
