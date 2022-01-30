import argparse
import os
import shutil

import tensorflow as tf
from third_party_tools.tool_gauss import generate_aux_video
from stylize import test
from train import train
from image_stylization.evaluate import transfer
from tools.preprocess_video import preprocess_video

STYLE_IMAGE = 'mosaic'
OUTPUT_DIR = 'output'

style_weight_path = 'image_stylization/weights'
training_y_path = 'training_y'
training_x_path = 'training_x'
flow_fwd_path = "flow_fwd"
flow_bwd_path = "flow_bwd"
aux_video_path = "input_gdisko_gauss_r10_s10"  # path to the result gauss r10 s15 sequence
keyframes_path = 'keyframes'

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Easy Style Transfer for Videos')
    parser.add_argument('video',
                        metavar='<video>',
                        help="Path to input video image sequence")
    parser.add_argument('--debug', required=False, type=bool,
                        metavar='<debug>',
                        help='Whether to print the loss',
                        default=False)
    parser.add_argument('--skip', required=False, type=int,
                        metavar='<skip>',
                        help='Steps to skip',
                        default=0)
    parser.add_argument('--style', required=True,
                        metavar=STYLE_IMAGE,
                        help='Style image to train the specific style',
                        default=STYLE_IMAGE)
    parser.add_argument('--output', required=False,
                        metavar='<output>',
                        help='Path to the transfer results',
                        default=OUTPUT_DIR)

    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    input_path = args.video

    frames = os.listdir(input_path)
    last_frame = sorted(frames)[-1]
    first_frame = sorted(frames)[0]
    file_ext = os.path.splitext(first_frame)[-1]
    last_frame = int(os.path.splitext(last_frame)[0])
    first_frame = int(os.path.splitext(first_frame)[0])

    if args.skip == 0:
        if os.path.exists(training_x_path):
            shutil.rmtree(training_x_path)
        os.mkdir(training_x_path)
        if os.path.exists(training_y_path):
            shutil.rmtree(training_y_path)
        os.mkdir(training_y_path)
        if os.path.exists(keyframes_path):
            shutil.rmtree(keyframes_path)
        os.mkdir(keyframes_path)
        if os.path.exists(aux_video_path):
            shutil.rmtree(aux_video_path)
        os.mkdir(aux_video_path)
        if os.path.exists(flow_fwd_path):
            shutil.rmtree(flow_fwd_path)
        os.mkdir(flow_fwd_path)
        if os.path.exists(flow_bwd_path):
            shutil.rmtree(flow_bwd_path)
        os.mkdir(flow_bwd_path)
        if not os.path.exists("generated"):
            os.mkdir("generated")

    step = 0
    if step >= args.skip:
        if args.debug:
            print('Computing bidirectional optical flow and extracting keyframes')
        preprocess_video(input_path)

    step += 1
    if step >= args.skip:
        if args.debug:
            print('Generating auxiliary video channels')
        generate_aux_video(first_frame, last_frame)

    step += 1
    if step >= args.skip:
        if args.debug:
            print('Transferring keyframes to target style')
        for keyframe in os.listdir(keyframes_path):
            if args.debug:
                print(f'Stylizing keyframe {keyframe}')
            shutil.copy(os.path.join(input_path, keyframe), os.path.join(training_x_path, keyframe))
            transfer(os.path.join(input_path, keyframe), os.path.join(style_weight_path, args.style, 'weights'),
                     None, os.path.join(training_y_path, keyframe))

    step += 1
    if step >= args.skip:
        if args.debug:
            print('Training the video stylization module')
        train(training_x_path, training_y_path, aux_video_path)

    step += 1
    if step >= args.skip:
        if args.debug:
            print('Stylizing complete video sequence')
        test(input_path, aux_video_path, args.output)
