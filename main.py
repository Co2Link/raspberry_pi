import tensorflow as tf
import argparse
import time
import cv2

from server.detector import Detector
from server.streamCapture import StreamCapture_http
from server.utils import Frame_rate_calculator

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--url',type=str,default='http://172.20.10.10:8000/stream.mjpg')
args = parser.parse_args()


def main():

    cap = StreamCapture_http(args.url)

    fc = Frame_rate_calculator()

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        while True:
            fc.start_record()
            input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            print(keypoint_coords.shape)
            print(pose_scores.shape)
            print(keypoint_scores.shape)

            # # # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            fc.frame_end()


            cv2.putText(overlay_image,'FPS:'+str(fc.get_frame_rate()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

            cv2.imshow('posenet', overlay_image)
            if cv2.waitKey(1) == 27:
                break

if __name__ == "__main__":
    main()