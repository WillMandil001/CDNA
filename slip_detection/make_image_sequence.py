# -*- coding: utf-8 -*-
### RUN IN PYTHON 3
import os
import cv2
import csv
import glob
import click
import logging


from PIL import Image 
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates


class DataFormatter():
    def __init__(self, data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction, upscale_image):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.sequence_length = sequence_length
        self.image_original_width = image_original_width
        self.image_original_height = image_original_height
        self.image_original_channel = image_original_channel
        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height
        self.state_action_dimension = state_action_dimension
        self.create_img = create_img
        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height
        self.create_img_prediction = create_img_prediction
        self.upscale_image = upscale_image
        self.image_original_width = image_original_width
        self.image_original_height = image_original_height


        self.logger = logging.getLogger(__name__)
        self.logger.info('making final data set from raw data')

        with tf.Session() as sess:
            files = glob.glob(data_dir + '/*')
            if len(files) == 0:
                self.logger.error("No files found with extensions .tfrecords in directory {0}".format(self.out_dir))
                exit()

            robot_pos_files = []
            for file in sorted(files):
                if file[0:55] == "/home/user/Robotics/slip_detection_franka/Dataset/robot":
                    robot_pos_files.append(file)

            # tactile_sensor_files = []
            # for file in sorted(files):
            #     if file[0:66] == "/home/user/Robotics/slip_detection_franka/Dataset/xelaSensor2_test":
            #         tactile_sensor_files.append(file)

            slip_labels_files = []
            for file in sorted(files):
                if file[0:55] == "/home/user/Robotics/slip_detection_franka/Dataset/label":
                    slip_labels_files.append(file)
            
            tactile_sensor_files = []
            for file in sorted(files):
                if file[0:68] == "/home/user/Robotics/slip_detection_franka/Dataset/xelaSensor2_manual":
                    tactile_sensor_files.append(file)

            robot_positions = []
            image_names = []
            image_names_labels = []
            slip_labels = []
            frequency_rate = 10
            images_for_viewing = []

            min_max_calc = []

            for i in range(1, 2):
                vals = np.asarray(pd.read_csv(tactile_sensor_files[i], header=None))[1:]
                slip_labels_sample = np.asarray(pd.read_csv(slip_labels_files[i], header=None)[1:])
                if slip_labels_sample[0][3] == '0.0':
                    for i in range(50, len(vals)):
                        min_max_calc.append(vals[i])
            self.min_max = self.find_min_max(min_max_calc)

            for i in range(1, data_set_length):
                images_new_sample = np.asarray(pd.read_csv(tactile_sensor_files[i], header=None))[1:]
                robot_positions_new_sample = np.asarray(pd.read_csv(robot_pos_files[i], header=None))
                robot_positions_files = np.asarray([robot_positions_new_sample[j*frequency_rate] for j in range(1, min(len(images_new_sample), int(len(robot_positions_new_sample)/frequency_rate)))])
                images_new_sample = images_new_sample[1:len(robot_positions_files)+1]
                slip_labels_sample = np.asarray(pd.read_csv(slip_labels_files[i], header=None)[1:])

                for j in range(10, len(robot_positions_files) - sequence_length):  # 1 IGNORES THE HEADER
                    robot_positions__ = []
                    images = []
                    images_labels = []
                    slip_labels_sample__ = []
                    for t in range(0, sequence_length):
                        robot_positions__.append(self.convert_to_state(robot_positions_files[j+t]))  # Convert from HTM to euler task space and quaternion orientation. [[was just [t]]]]
                        images.append(self.create_image(images_new_sample[j+t]))  # [video location, frame]
                        images_labels.append(images_new_sample[j+t+1])  # [video location, frame]
                        slip_labels_sample__.append(slip_labels_sample[j+t][2])

                    robot_positions.append([state for state in robot_positions__])  # this works for testing their system
                    image_names.append(images)
                    image_names_labels.append(images_labels)
                    slip_labels.append(slip_labels_sample__)

                if slip_labels_sample[0][3] != '0.0':
                    self.view_iamge_sequence(images_new_sample, slip_labels_sample)

            print(aaa)

            self.slip_labels = np.asarray(slip_labels)
            self.robot_positions = np.asarray(robot_positions)
            self.image_names = np.asarray(image_names)
            self.image_names_labels = np.asarray(image_names_labels)

            self.csv_ref = []
            self.process_data()
            self.save_data_to_map()

    def view_iamge_sequence(self, image_list, slip_labels_sample):
        slips = []
        images = []
        for image_name, slip_label in zip(image_list, slip_labels_sample):
            images.append(self.create_image(image_name)*255)
            slips.append(slip_label[2])

        for index, (image, slip) in enumerate(zip(images, slips)):
            # single channel:
            image = image.reshape(3, self.image_resize_width, self.image_resize_height)[0]
            image = image.reshape(self.image_resize_width, self.image_resize_height, 1)
            cv2.imwrite("/home/user/Robotics/CDNA/images/manual_slip_normal/image_step" + str(index) + "slip_" + str(slip) + ".png", image)

            # RGB image:
            # im = Image.fromarray(np.uint8(image)).convert('RGB')
            # im.save("/home/user/Robotics/CDNA/images/manual_slip/image_step" + str(index) + "slip_" + str(slip) + ".jpeg")

        print(aaaa)

    def find_min_max(self, imlist):
        for values in imlist:
            vals = np.asarray(values).astype(float).reshape(4,4,3)
            try:
                normal_min = min([min(vals[:, :, 0].flatten()), normal_min])
                normal_max = max([max(vals[:, :, 0].flatten()), normal_max])
                sheerx_min = min([min(vals[:, :, 1].flatten()), sheerx_min])
                sheerx_max = max([max(vals[:, :, 1].flatten()), sheerx_max])
                sheery_min = min([min(vals[:, :, 2].flatten()), sheery_min])
                sheery_max = max([max(vals[:, :, 2].flatten()), sheery_max])
            except:
                normal_min = min(vals[:, :, 0].flatten())
                normal_max = max(vals[:, :, 0].flatten())
                sheerx_min = min(vals[:, :, 1].flatten())
                sheerx_max = max(vals[:, :, 1].flatten())
                sheery_min = min(vals[:, :, 2].flatten())
                sheery_max = max(vals[:, :, 2].flatten())

        return [normal_min, normal_max, sheerx_min, sheerx_max, sheery_min, sheery_max]

    def create_image(self, image_raw):
        image = np.asarray(image_raw).astype(float)
        image = image.reshape(self.image_original_width, self.image_original_height,3)
        for x in range(0, len(image[0])):
            for y in range(0, len(image[1])):
                image[x][y][0] = ((image[x][y][0] - self.min_max[0]) / (self.min_max[1] - self.min_max[0])) * 255  # Normalise normal
                image[x][y][1] = ((image[x][y][1] - self.min_max[2]) / (self.min_max[3] - self.min_max[2])) * 255  # Normalise shearx
                image[x][y][2] = ((image[x][y][2] - self.min_max[4]) / (self.min_max[5] - self.min_max[4])) * 255  # Normalise sheary

        if self.upscale_image:
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            image = image_pil.resize((self.image_resize_width, self.image_resize_height)) # , Image.ANTIALIAS)
            image = np.asarray(image)

        return (image.astype(np.float32) / 255.0)

    def convert_to_state(self, pose):
        state = [pose[16], pose[17], pose[18]]
        return state


@click.command()
@click.option('--data_set_length', type=click.INT, default=70, help='size of dataset to format.')
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/slip_detection_franka/Dataset/', help='Directory containing data.')  # /home/user/Robotics/Data_sets/data_set_003/
@click.option('--out_dir', type=click.Path(), default='/home/user/Robotics/Data_sets/CDNA_data/32x32_tactile', help='Output directory of the converted data.')
@click.option('--sequence_length', type=click.INT, default=10, help='Sequence length, including context frames.')
@click.option('--image_original_width', type=click.INT, default=4, help='Original width of the images.')
@click.option('--image_original_height', type=click.INT, default=4, help='Original height of the images.')
@click.option('--image_original_channel', type=click.INT, default=3, help='Original channels amount of the images.')
@click.option('--image_resize_width', type=click.INT, default=32, help='Resize width of the the images.')
@click.option('--image_resize_height', type=click.INT, default=32, help='Resize height of the the images.')
@click.option('--state_action_dimension', type=click.INT, default=5, help='Dimension of the state and action.')
@click.option('--create_img', type=click.INT, default=0, help='Create the bitmap image along the numpy RGB values')
@click.option('--create_img_prediction', type=click.INT, default=1, help='Create the bitmap image used in the prediction phase')
@click.option('--upscale_image', type=click.INT, default=1, help='Upscale the image to a new dimension?')
def main(data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction, upscale_image):
    data_formatter = DataFormatter(data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction, upscale_image)

if __name__ == '__main__':
    main()
