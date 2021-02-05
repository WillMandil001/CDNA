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

            tactile_sensor_files = []
            for file in sorted(files):
                if file[0:66] == "/home/user/Robotics/slip_detection_franka/Dataset/xelaSensor1_test":
                    tactile_sensor_files.append(file)

            slip_labels_files = []
            for file in sorted(files):
                if file[0:61] == "/home/user/Robotics/slip_detection_franka/Dataset/labels_test":
                    slip_labels_files.append(file)

            robot_positions = []
            image_names = []
            image_names_labels = []
            slip_labels = []
            frequency_rate = 10
            images_for_viewing = []

            min_max_calc = []

            for i in range(1, data_set_length):
                images_new_sample = np.asarray(pd.read_csv(tactile_sensor_files[i], header=None))[1:]
                robot_positions_new_sample = np.asarray(pd.read_csv(robot_pos_files[i], header=None))
                robot_positions_files = np.asarray([robot_positions_new_sample[j*frequency_rate] for j in range(1, min(len(images_new_sample), int(len(robot_positions_new_sample)/frequency_rate)))])
                images_new_sample = images_new_sample[1:len(robot_positions_files)+1]
                slip_labels_sample = np.asarray(pd.read_csv(slip_labels_files[i], header=None)[1:])

                for j in range(0, len(robot_positions_files) - sequence_length):  # 1 IGNORES THE HEADER
                    robot_positions__ = []
                    images = []
                    images_labels = []
                    slip_labels_sample__ = []
                    for t in range(0, sequence_length):
                        robot_positions__.append(self.convert_to_state(robot_positions_files[j+t]))  # Convert from HTM to euler task space and quaternion orientation. [[was just [t]]]]
                        images.append(self.create_movement_image(images_new_sample[j+t], images_new_sample[0], color=(255,255,255)))  # [video location, frame]
                        images_labels.append(images_new_sample[j+t+1])  # [video location, frame]
                        slip_labels_sample__.append(slip_labels_sample[j+t][2])

                    robot_positions.append([state for state in robot_positions__])  # this works for testing their system
                    image_names.append(images)
                    image_names_labels.append(images_labels)
                    slip_labels.append(slip_labels_sample__)

                # if slip_labels_sample[0][3] != '0.0':
                #     self.view_iamge_sequence(images_new_sample, slip_labels_sample)

            print(aaa)

            self.slip_labels = np.asarray(slip_labels)
            self.robot_positions = np.asarray(robot_positions)
            self.image_names = np.asarray(image_names)
            self.image_names_labels = np.asarray(image_names_labels)

            self.csv_ref = []
            self.process_data()
            self.save_data_to_map()


    def create_movement_image(self, data, base, color):
        width  = 800 # width of the image
        height = 800 # height of the image
        margin = 160 # margin of the taxel in the image
        scale = 15
        radius = 15
        img = np.zeros((height,width,3), np.uint8)
        cv2.namedWindow('xela-sensor', cv2.WINDOW_NORMAL)

        diff = np.array(data) - np.array(base)
        diff = diff.reshape(4,4,3)
        diff = diff.T.reshape(3,4,4)
        dx = np.rot90((np.flip(diff[0], axis=0) / scale), k=3, axes=(0,1)).flatten()
        dy = np.rot90((np.flip(diff[1], axis=0) / scale), k=3, axes=(0,1)).flatten()
        dz = np.rot90((np.flip(diff[2], axis=0) / scale), k=3, axes=(0,1)).flatten()

        image_positions = []
        for x in range(margin, 5*margin, margin):
            for y in range(margin, 5*margin, margin):
                image_positions.append([y,x])

        for xx, yy ,zz, image_position in zip(dx, dy, dz, image_positions):
            z = radius # + (abs(xx))
            x = image_position[0] + int(-xx) # int(zz)
            y = image_position[1] + int(-yy) # int(yy)
            # color = (255-int(z/100 * 255), 210, 255-int(z/100 * 255))  # Draw sensor circles
            cv2.circle(img, (x, y), int(z), color=color, thickness=-1)  # Draw sensor circles
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        print(gray)
        print(aaa)
        return gray

    def visualise_time_sequence(self, data, base, color):
        # tactile_sensor_files1 = "/home/user/Robotics/slip_detection_franka/Dataset/xela_validation/xelaSensor1_left2right.csv"  #  xelaSensor1_bottomup   xelaSensor1_left2right.csv"
        # tactile_sensor_files2 = "/home/user/Robotics/slip_detection_franka/Dataset/xela_validation/xelaSensor1_right2left.csv"
        # tactile_sensor_files3 = "/home/user/Robotics/slip_detection_franka/Dataset/xela_validation/xelaSensor1_topdown.csv"
        # tactile_sensor_files4 = "/home/user/Robotics/slip_detection_franka/Dataset/xela_validation/xelaSensor1_bottomup.csv"
        # tactile_sensor_files5 = "/home/user/Robotics/slip_detection_franka/Dataset/xela_validation/xelaSensor1_spin.csv"

        # images_new_sample1 = np.asarray(pd.read_csv(tactile_sensor_files1, header=None))[1:]
        # images_new_sample2 = np.asarray(pd.read_csv(tactile_sensor_files2, header=None))[1:]
        # images_new_sample3 = np.asarray(pd.read_csv(tactile_sensor_files3, header=None))[1:]
        # images_new_sample4 = np.asarray(pd.read_csv(tactile_sensor_files4, header=None))[1:]
        # images_new_sample5 = np.asarray(pd.read_csv(tactile_sensor_files5, header=None))[1:]
        # image_set = [images_new_sample1, images_new_sample2, images_new_sample3, images_new_sample4, images_new_sample5]
        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # while 1:
        #     for color, images in zip(colors, image_set):
        #         for image in images:
        #             base  = image.astype(np.float32)
        #             image = images_new_sample1[0].astype(np.float32)
        #             self.visualise_time_sequence(image, base, color)
        width  = 800 # width of the image
        height = 800 # height of the image
        margin = 160 # margin of the taxel in the image
        scale = 15
        radius = 15

        img = np.zeros((height,width,3), np.uint8)
        cv2.namedWindow('xela-sensor', cv2.WINDOW_NORMAL)

        diff = np.array(data) - np.array(base)
        diff = diff.reshape(4,4,3)
        diff = diff.T.reshape(3,4,4)
        dx = np.rot90((np.flip(diff[0], axis=0) / scale), k=3, axes=(0,1)).flatten()
        dy = np.rot90((np.flip(diff[1], axis=0) / scale), k=3, axes=(0,1)).flatten()
        dz = np.rot90((np.flip(diff[2], axis=0) / scale), k=3, axes=(0,1)).flatten()

        image_positions = []
        for x in range(margin, 5*margin, margin):
            for y in range(margin, 5*margin, margin):
                image_positions.append([y,x])

        for xx, yy ,zz, image_position in zip(dx, dy, dz, image_positions):
            z = radius # + (abs(xx))
            x = image_position[0] + int(-xx) # int(zz)
            y = image_position[1] + int(-yy) # int(yy)
            # color = (255-int(z/100 * 255), 210, 255-int(z/100 * 255))  # Draw sensor circles
            cv2.circle(img, (x, y), int(z), color=color, thickness=-1)  # Draw sensor circles

        cv2.imshow("xela-sensor", img)   # Display sensor image
        key = cv2.waitKey(40)
        if key == 27:
            cv2.destroyAllWindows()


    def view_iamge_sequence(self, image_list, slip_labels_sample):
        slips = []
        images = []
        for image_name, slip_label in zip(image_list, slip_labels_sample):
            images.append(self.create_image(image_name)*255)
            slips.append(slip_label[2])

        for index, (image, slip) in enumerate(zip(images, slips)):
            image = image.reshape(3, self.image_resize_width, self.image_resize_height)[2]
            image = image.reshape(self.image_resize_width, self.image_resize_height, 1)
            cv2.imwrite("/home/user/Robotics/CDNA/images/sheary/image_step" + str(index) + "slip_" + str(slip) + ".png", image)
            # im = Image.fromarray(np.uint8(image)).convert('RGB')
            # im.save("/home/user/Robotics/CDNA/images/set1/image_step" + str(index) + "slip_" + str(slip) + ".jpeg")

        print(aaaa)

    # def find_min_max(self, imlist):
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
            image = image_pil.resize((self.image_resize_width, self.image_resize_height), Image.ANTIALIAS)
            image = np.asarray(image)

        return (image.astype(np.float32) / 255.0)

    def convert_to_state(self, pose):
        state = [pose[16], pose[17], pose[18]]
        return state

    def process_data(self):
        for j in tqdm(range(0, int(len(self.image_names) / 2))):
        # for j in tqdm(range(0, int(32*20))):
            raw = []
            for k in range(len(self.image_names[j])):
                tmp = self.image_names[j][k].astype(np.float32)
                raw.append(tmp)
            raw = np.array(raw)

            ref = []
            ref.append(j)

            ### save png data
            if self.create_img == 1:
                for k in range(raw.shape[0]):
                    img = Image.fromarray(raw[k], 'RGB')
                    img.save(self.out_dir + '/image_batch_' + str(j) + '_' + str(k) + '.png')
                ref.append('image_batch_' + str(j) + '_*' + '.png')
            else:
                ref.append('')

            ### save np images
            np.save(self.out_dir + '/image_batch_' + str(j), raw)

            ### save np action
            np.save(self.out_dir + '/action_batch_' + str(j), self.robot_positions[j])

            ### save np states
            np.save(self.out_dir + '/state_batch_' + str(j), self.robot_positions[j])  # original

            ### save np images
            np.save(self.out_dir + '/slip_label_batch_' + str(j), self.slip_labels[j])

            # save names for map file
            ref.append('image_batch_' + str(j) + '.npy')
            ref.append('action_batch_' + str(j) + '.npy')
            ref.append('state_batch_' + str(j) + '.npy')

            # Image used in prediction
            if self.create_img_prediction == 1:
                pred = []
                for k in range(len(self.image_names[j])):
                    img = Image.fromarray(self.image_names[j][k], 'RGB')  # [1]
                    img.save(self.out_dir + '/image_batch_pred_' + str(j) + '_' + str(k) + '.png')
                    pred.append(np.array(img))
                np.save(self.out_dir + '/image_batch_pred_' + str(j), np.array(pred))
                ref.append('image_batch_pred_' + str(j) + '_*' + '.png')
                ref.append('image_batch_pred_' + str(j) + '.npy')
            else:
                ref.append('')
                ref.append('')

            ref.append('slip_label_batch_' + str(j) + '.npy')

            ### Append all file names for this sample to CSV file for training.
            self.csv_ref.append(ref)

    def getFrame(self, sec, vidcap):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        return image

    def save_data_to_map(self):
        self.logger.info("Writing the results into map file '{0}'".format('map.csv'))
        with open(self.out_dir + '/map.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['id', 'img_bitmap_path', 'img_np_path', 'action_np_path', 'state_np_path', 'img_bitmap_pred_path', 'img_np_pred_path', 'slip_label'])
            for row in self.csv_ref:
                writer.writerow(row)


@click.command()
@click.option('--data_set_length', type=click.INT, default=70, help='size of dataset to format.')
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/slip_detection_franka/Dataset/', help='Directory containing data.')  # xela_validation/ /home/user/Robotics/Data_sets/data_set_003/
@click.option('--out_dir', type=click.Path(), default='/home/user/Robotics/Data_sets/CDNA_data/asdf', help='Output directory of the converted data.')
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

    # a = data
    # print(a)
    # print("======")
    # a = a.reshape(4,4,3)
    # print(a)
    # print("======")
    # a = a.T.reshape(3,4,4)
    # print(a)
    # print("======")
    # b = np.flip(a[0], axis=0)
    # c = np.flip(a[1], axis=0)
    # d = np.flip(a[2], axis=0)
    # print(b)
    # print(c)
    # print(d)
    # print("======")
    # b = np.rot90(b, k=3, axes=(0,1))
    # c = np.rot90(c, k=3, axes=(0,1))
    # d = np.rot90(d, k=3, axes=(0,1))
    # print(b)
    # print(c)
    # print(d)
    # print("======")
    # b = b.flatten()
    # c = c.flatten()
    # d = d.flatten()
    # print(b)
    # print(c)
    # print(d)
    # print("======")