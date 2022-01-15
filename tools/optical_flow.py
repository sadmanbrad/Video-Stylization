import sys

import cv2
import numpy as np


def dense_optical_flow(method, video_path, image_path, params=[], to_gray=False):
    frame_index = 0
    # Read the video and first frame
    old_frame = cv2.imread(video_path + f'{frame_index:04}.png')

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    prev_frame = cv2.imread(image_path)

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        frame_index += 1
        # Read the next frame
        new_frame = cv2.imread(video_path + f'{frame_index:04}.png')
        frame_copy = new_frame
        if new_frame is None:
            break

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)
        height, width = flow.shape[0], flow.shape[1]
        R2 = np.dstack(np.meshgrid(np.arange(float(width)), np.arange(float(height))))
        print(R2.shape, flow.shape)
        pixel_map = R2 - flow
        xs = pixel_map[:, :, 0].astype(np.float32)
        ys = pixel_map[:, :, 1].astype(np.float32)
        new_image = cv2.remap(prev_frame, xs, ys, cv2.INTER_LINEAR)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        cv2.imshow("optical flow", new_image)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame
        old_frame = new_frame
        prev_frame = new_image


if __name__ == '__main__':
    dense_optical_flow(cv2.cv2.DISOpticalFlow_create(cv2.cv2.DISOpticalFlow_PRESET_MEDIUM).calc, sys.argv[1], sys.argv[2], [], True)
