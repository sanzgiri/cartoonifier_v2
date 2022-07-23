# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import json
import os
import cv2
import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
import tensorflow.compat.v1 as tf
import tf_slim as slim

from pathlib import Path
import time
from datetime import timedelta
import tempfile
from cog import BasePredictor, Input, Path

tf.disable_v2_behavior()


def tf_box_filter(x, r):
    k_size = int(2 * r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size ** 2)
    box_kernel = weight * np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)
    # y_shape = tf.shape(y)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    # assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4

    lr_x_shape = tf.shape(lr_x)
    # lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)

    N = tf_box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.image.resize_images(A, hr_x_shape[1: 3])
    mean_b = tf.image.resize_images(b, hr_x_shape[1: 3])

    output = mean_A * hr_x + mean_b

    return output


def resblock(inputs, out_channel=32, name='resblock', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = slim.convolution2d(inputs, out_channel, [3, 3],
                               activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3],
                               activation_fn=None, scope='conv2')

        return x + inputs


def unet_generator(inputs, channel=32, num_blocks=4, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)

        return x4


def resize_crop(image, dimension_min=720):
    h, w, c = np.shape(image)
    if min(h, w) > dimension_min:
        if h > w:
            h, w = int(dimension_min * h / w), dimension_min
        else:
            h, w = dimension_min, int(dimension_min * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def video_stats(cap):
    """Gets video stats. OpenCV is not reliable supply us with total frame count or milliseconds,
    so we have to manually calculate them by iterating over the frames ourselves.
    Parameters
    ----------
    cap : VideoCapture
        object for video capturing from video files

    Returns
    -------
    total_milliseconds
        total video duration in milliseconds
    total_frames
        total frame count from video duration
    """

    total_milliseconds = 0
    total_frames = 0

    # jump to the end of the video and get the current milliseconds and frame number
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    total_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
    total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # reset to frame zero
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    print(f"[INFO] total milliseconds: {total_milliseconds}")
    print(f"[INFO] total frames: {total_frames}")

    return total_milliseconds, total_frames


def cartoonize_video(content_path, output_path, sess, input_photo, final_out,
                     time_interval, dimension_min, verbose=True):
    """Gets all results from running inference on images in the entire video file
    Parameters
    ----------
    content_path : str
        The local path the the video file we want to analyze
    output_path : str
        The output path for the target video
    sess : tensorflow session
    input_photo : tensorflow placeholder image
    final_out : tensorflow output image
    time_interval : float, optional
        how often do we want to sample (e.g. time between sample)
    dimension_min : int, min size for either width or height
    verbose: bool
        verbose printing of processing at regular intervals
    Returns
    -------
    array
        an array of objects representing keypoints identified in the video frames
    """

    # use OpenCV to open video file
    cap = cv2.VideoCapture(content_path)
    print(f"[INFO] asset: {content_path}")

    # get total milliseconds
    total_milliseconds, total_frames = video_stats(cap)

    # images_per_second = 5 #
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    PRINT_INTERVAL = fps * 5
    time_print_next = 0  # next print time
    cumulative_inference = 0

    # get temp item
    obj_temp_in = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    obj_temp_in.close()
    obj_temp_out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    obj_temp_out.close()

    hop = round(fps * time_interval)
    print(f"Info: Detected HOP interval {hop} from fps {fps} and time_interval {time_interval}")
    out = None

    # holds all results from inference call
    results = []
    while (cap.isOpened()):

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if we've reached the end of the video
        if not hasFrame:
            time.sleep(3)

            # Release device
            cap.release()
            if out is not None:
                out.release()
            break

        # get current milliseconds and frame number
        milliseconds = cap.get(0)
        frame_number = int(cap.get(1))

        if hop == 0 or (frame_number % hop == 0):
            start_inference = cv2.getTickCount()
            source_size = (frame.shape[1], frame.shape[0])

            # run inference on image from video
            frame_adj = cartoonize(frame, sess, input_photo, final_out, dimension_min=dimension_min)

            # create the output video if we need to
            if out is None:
                output_size = (frame_adj.shape[1], frame_adj.shape[0])
                path_output = str(Path(output_path).with_suffix(".mp4"))
                target_fps = fps if time_interval == 0 else 1 / time_interval
                print(
                    f"New video: {path_output}, target fps: {target_fps}, original_size: {source_size}, video_size: {output_size}, min_dimension: {dimension_min}")
                out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, output_size)

            # write the new video
            out.write(frame_adj)
            # final time for inferring these frames
            complete_inference = cv2.getTickCount()
            time_in_sec = round(milliseconds / float(1000), 3)

            # total inference time for the frame
            inference_time = (complete_inference - start_inference) / cv2.getTickFrequency()
            cumulative_inference += inference_time
            if milliseconds > time_print_next:
                time_print_next += PRINT_INTERVAL
                if verbose:
                    time_avg = cumulative_inference / frame_number
                    time_eta = time_avg * total_frames
                    time_ellapse = str(timedelta(seconds=cumulative_inference)).split('.')[0]
                    time_remain = str(timedelta(seconds=time_eta - cumulative_inference)).split('.')[0]
                    print(
                        f"[INFO] video {time_in_sec}s | frame: {frame_number}/{total_frames} | process: {round(inference_time, 2)}s | proc_remain: {time_remain} | proc_ellapsed: {time_ellapse}")

    # delete temp file used for writing frames
    if obj_temp_in is not None:
        path_temp = Path(obj_temp_in.name)
        if path_temp.exists():
            path_temp.unlink()
    if obj_temp_out is not None:
        path_temp = Path(obj_temp_out.name)
        if path_temp.exists():
            path_temp.unlink()

    return results


def cartoonize(input_nd, sess, input_photo, final_out, dimension_min):
    input_nd = resize_crop(input_nd, dimension_min)
    batch_image = input_nd.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output_nd = sess.run(final_out, feed_dict={input_photo: batch_image})
    output_nd = (np.squeeze(output_nd) + 1) * 127.5
    output_nd = np.clip(output_nd, 0, 255).astype(np.uint8)
    return output_nd


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(self,
                infile: Path = Input(description="Input video"),
                frame_rate: int = Input(description="Frames per second to sample", default=24),
                horizontal_resolution: int = Input(description="Horizontal video resolution", default=480)
                ) -> Path:

        """Run a single prediction on the model"""
        print(f"File: {infile}")
        model_path = 'saved_models'
        time_interval = float(1/frame_rate)
        dimension_min = horizontal_resolution

        tf.reset_default_graph()
        input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
        network_out = unet_generator(input_photo)
        final_out = guided_filter(input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        #infile_path = Path(infile)
        outfile = "./processed.mp4"

        cap = cv2.VideoCapture(str(infile))
        total_milliseconds, total_frames = video_stats(cap)
        print(f"File: {infile}, time: {total_milliseconds}, frames: {total_frames}, outfile: {outfile}")

        os.system("rm -f ./output_audio.mp3")
        os.system("rm -f ./output_video.mp4")
        os.system("rm -f ./output_cartoon.mp4")
        os.system("rm -f ./processed.mp4")

        audiofile = f"./output_audio.mp3"
        videofile = f"./output_video.mp4"
        mergedfile = f"./output_cartoon.mp4"
        split_cmd = f"ffmpeg -i {infile} -map 0:a {audiofile} -map 0:v {videofile}"
        print(split_cmd)
        os.system(split_cmd)

        cartoonize_video(videofile, outfile, sess, input_photo, final_out,
                        time_interval, dimension_min, verbose=True)

        merge_cmd = f"ffmpeg -i {outfile} -i {audiofile} {mergedfile}"
        print(merge_cmd)
        os.system(merge_cmd)

        return Path(mergedfile)


