#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Predict the next n frames from a trained model
# ==============================================

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from matplotlib import pyplot as plt

from model import Model
from model import concat_examples

try:
    import cupy
except:
    cupy = np
    pass

import click
import os
import csv
import logging
import glob
import subprocess

import six.moves.cPickle as pickle

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops
import imageio

# ========================
# Helpers functions (hlpr)
# ========================

def get_data_info(data_dir, data_index, get_changing_data):
    data_map = []
    with open(data_dir + '/map.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data_map.append(row)

    if len(data_map) <= 1: # empty or only header
        raise ValueError("No file map found")

    # Get the requested data to test
    breaker = 0
    if get_changing_data == 1:
        for i in range(1, len(data_map)):
            image = np.float32(np.load(data_dir + '/' + data_map[i][2]))
            for j in range(0, len(image) - 1):
                if np.sum(image[j] - image[j+1]) * 255 > 10 or np.sum(image[j] - image[j+1]) * 255 < -10:
                    data_index = i
                    print(i, )
                    breaker = 1
            if breaker:
                break

    data_index = int(data_index)+1
    if data_index > len(data_map)-1:
        raise ValueError("Data index {} is out of range for available data".format(data_index))

    # print(data_map[data_index])

    image = np.float32(np.load(data_dir + '/' + data_map[data_index][2]))
    image_pred = np.float32(np.load(data_dir + '/' + data_map[data_index][6]))
    image_bitmap_pred = data_map[data_index][5]
    action = np.float32(np.load(data_dir + '/' + data_map[data_index][3]))
    state = np.float32(np.load(data_dir + '/' + data_map[data_index][4]))

    # from matplotlib import pyplot as plt
    # for i in image:
    #     plt.imshow(i)
    #     plt.show(block=False)
    #     plt.pause(0.5)

    return image, image_pred, image_bitmap_pred, action, state

# =================================================
# Main entry point of the training processes (main)
# =================================================
@click.command()
@click.option('--model_dir', type=click.STRING, default='20210124-143736-CDNA-16', help='Directory containing model.')
@click.option('--model_name', type=click.STRING, default='training-30', help='The name of the model.')
@click.option('--data_index', type=click.INT, default=20, help='Directory containing data.')
@click.option('--get_changing_data', type=click.INT, default=1, help='Should the program look for a test sample where there is change in the time step.')
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/CDNA_data/4x4_tactile', help='Directory containing data.')
@click.option('--time_step', type=click.INT, default=10, help='Number of time steps to predict.')
@click.option('--model_type', type=click.STRING, default='CDNA', help='Type of the trained model.')
@click.option('--schedsamp_k', type=click.FLOAT, default=-1, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--context_frames', type=click.INT, default=2, help='Number of frames before predictions.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--image_height', type=click.INT, default=4, help='Height of one predicted frame.')
@click.option('--image_width', type=click.INT, default=4, help='Width of one predicted frame.')
@click.option('--original_image_height', type=click.INT, default=4, help='Height of one predicted frame.')
@click.option('--original_image_width', type=click.INT, default=4, help='Width of one predicted frame.')
@click.option('--downscale_factor', type=click.FLOAT, default=1, help='Downscale the image by this factor. (was 0.5)')
@click.option('--gpu', type=click.INT, default=0, help='ID of the gpu to use')
@click.option('--gif', type=click.INT, default=1, help='Create a GIF of the predicted result.')
def main(model_dir, model_name, data_index, get_changing_data, models_dir, data_dir, time_step, model_type, schedsamp_k, context_frames, use_state, num_masks, image_height, image_width, original_image_height, original_image_width, downscale_factor, gpu, gif):
    """ Predict the next {time_step} frame based on a trained {model} """
    logger = logging.getLogger(__name__)
    path = models_dir + '/' + model_dir
    if not os.path.exists(path + '/' + model_name):
        raise ValueError("Directory {} does not exists".format(path))
    if not os.path.exists(data_dir):
        raise ValueError("Directory {} does not exists".format(data_dir))

    logger.info("Loading data {}".format(data_index))
    image, image_pred, image_bitmap_pred, action, state = get_data_info(data_dir, data_index, get_changing_data)
    
    img_pred, act_pred, sta_pred = concat_examples([[image, action, state]])

    # Extract the information about the model
    if model_type == '':
        split_name = model_dir.split("-")
        if len(split_name) != 4:
            raise ValueError("Model {} is not recognized, use --model_type to describe the type".format(model_dir))
        model_type = split_name[2]

    # Load the model for prediction
    logger.info("Importing model {0}/{1} of type {2}".format(model_dir, model_name, model_type))
    model = Model(
        num_masks=num_masks,
        is_cdna=model_type,
        use_state=use_state,
        scheduled_sampling_k=schedsamp_k,
        num_frame_before_prediction=context_frames,
        prefix='predict'
    )

    chainer.serializers.load_npz(str(path + '/' + model_name), model)
    logger.info("Model imported successfully")

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # Resize the image to fit the trained dimension
    resize_img_pred = img_pred

    # Predict the new images
    with chainer.using_config('train', False):
        loss = model(cupy.asarray(resize_img_pred, dtype=cupy.float32),
                     cupy.asarray(act_pred, dtype=cupy.float32),
                     cupy.asarray(sta_pred, dtype=cupy.float32),
                     0)
        predicted_images = model.gen_images
    
    f, axarr = plt.subplots(2,9)
    axarr[0,0].imshow(cupy.asnumpy(predicted_images[0].data[0] * 255).astype(np.uint8).T)
    axarr[0,0].set_title("0: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[0].data[0] * 255).astype(np.uint8).T))))
    axarr[0,1].imshow(cupy.asnumpy(predicted_images[1].data[0] * 255).astype(np.uint8).T)
    axarr[0,1].set_title("1: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[1].data[0] * 255).astype(np.uint8).T))))
    axarr[0,2].imshow(cupy.asnumpy(predicted_images[2].data[0] * 255).astype(np.uint8).T)
    axarr[0,2].set_title("2: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[2].data[0] * 255).astype(np.uint8).T))))
    axarr[0,3].imshow(cupy.asnumpy(predicted_images[3].data[0] * 255).astype(np.uint8).T)
    axarr[0,3].set_title("3: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[3].data[0] * 255).astype(np.uint8).T))))
    axarr[0,4].imshow(cupy.asnumpy(predicted_images[4].data[0] * 255).astype(np.uint8).T)
    axarr[0,4].set_title("4: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[4].data[0] * 255).astype(np.uint8).T))))
    axarr[0,5].imshow(cupy.asnumpy(predicted_images[5].data[0] * 255).astype(np.uint8).T)
    axarr[0,5].set_title("5: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[5].data[0] * 255).astype(np.uint8).T))))
    axarr[0,6].imshow(cupy.asnumpy(predicted_images[6].data[0] * 255).astype(np.uint8).T)
    axarr[0,6].set_title("6: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[6].data[0] * 255).astype(np.uint8).T))))
    axarr[0,7].imshow(cupy.asnumpy(predicted_images[7].data[0] * 255).astype(np.uint8).T)
    axarr[0,7].set_title("7: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[7].data[0] * 255).astype(np.uint8).T))))
    axarr[0,8].imshow(cupy.asnumpy(predicted_images[8].data[0] * 255).astype(np.uint8).T)
    axarr[0,8].set_title("8: P: {0}".format(int(np.sum(cupy.asnumpy(predicted_images[8].data[0] * 255).astype(np.uint8).T))))
    axarr[0,0].set_ylabel("Pred", rotation=90, size='large')

    axarr[1,0].imshow(image[0]*255)
    axarr[1,0].set_title("0: GT: {0}".format(int(np.sum(image[0]*255))))
    axarr[1,1].imshow(image[1]*255)
    axarr[1,1].set_title("1: GT: {0}".format(int(np.sum(image[1]*255))))
    axarr[1,2].imshow(image[2]*255)
    axarr[1,2].set_title("2: GT: {0}".format(int(np.sum(image[2]*255))))
    axarr[1,3].imshow(image[3]*255)
    axarr[1,3].set_title("3: GT: {0}".format(int(np.sum(image[3]*255))))
    axarr[1,4].imshow(image[4]*255)
    axarr[1,4].set_title("4: GT: {0}".format(int(np.sum(image[4]*255))))
    axarr[1,5].imshow(image[5]*255)
    axarr[1,5].set_title("5: GT: {0}".format(int(np.sum(image[5]*255))))
    axarr[1,6].imshow(image[6]*255)
    axarr[1,6].set_title("6: GT: {0}".format(int(np.sum(image[6]*255))))
    axarr[1,7].imshow(image[7]*255)
    axarr[1,7].set_title("7: GT: {0}".format(int(np.sum(image[7]*255))))
    axarr[1,8].imshow(image[8]*255)
    axarr[1,8].set_title("8: GT: {0}".format(int(np.sum(image[8]*255))))
    axarr[1,0].set_ylabel("GT", rotation=90, size='large')

    plt.show()

    print(aa)





    # Resize the predicted image
    resize_predicted_images = []
    for i in xrange(len(predicted_images)):
        resize = predicted_images[i].data[0]
        resize -= resize.min()
        resize /= resize.max()
        resize *= 255.0
        resize_predicted_images.append(cupy.asnumpy(resize.astype(np.uint8)))


    # Print the images horizontally
    # First row is the time_step
    # Second row is the ground truth
    # Third row is the generated image
    frame_width = int(original_image_width * downscale_factor)
    frame_height = int(original_image_height * downscale_factor)
    text_width_x = frame_width
    text_height_x = 50
    text_width_y = frame_height
    text_height_y = 50

    total_width = frame_width * time_step + text_height_x
    total_height = frame_height * 2 + text_height_x

    if gif == 1:
        total_width = total_width + frame_width

    new_image = Image.new('RGBA', (total_width, total_height))
    pred_image = Image.new('RGBA', (frame_width, frame_height))
    orig_image = Image.new('RGBA', (frame_width, frame_height))

    # Text label x
    font_size = 20
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size, encoding="unic")
    label = ["Time = {}".format(i+1) for i in xrange(time_step)]

    if gif == 1:
        label.append("GIF")

    for i in xrange(len(label)):
        text = label[i]
        text_container_img = Image.new('RGB', (text_width_x, text_height_x), 'black')
        text_container_draw = ImageDraw.Draw(text_container_img)
        w, h = text_container_draw.textsize(text, font=font)
        text_container_draw.text(((text_width_x-w)/2, (text_height_x-h)/2), text, fill='red', font=font)
        new_image.paste(text_container_img, (text_height_x + text_width_x*i, 0))

    # Text label y
    label = ["GT", "Pred"]
    for i in xrange(len(label)):
        text = label[i]
        text_container_img = Image.new('RGB', (text_width_y, text_height_y), 'black')
        text_container_draw = ImageDraw.Draw(text_container_img)
        w, h = text_container_draw.textsize(text, font=font)
        text_container_draw.text(((text_width_y-w)/2, (text_height_y-h)/2), text, fill='red', font=font)
        text_container_img = text_container_img.rotate(90, expand=1)
        new_image.paste(text_container_img, (0, text_height_x + text_width_y * i))

    # Original
    # ground_truth_images_path = glob.glob(data_dir + '/' + image_bitmap_pred)
    ground_truth_images_path = sorted(glob.glob(data_dir + '/' + image_bitmap_pred))
    original_gif = []
    original_images_np = []
    for i in xrange(min(time_step, len(ground_truth_images_path))):
        img = Image.open(ground_truth_images_path[i]).convert('RGB')

        if downscale_factor != 1:
            img = img.resize((frame_width, frame_height), Image.ANTIALIAS)
        original_images_np.append(np.asarray(img))
        new_image.paste(img, (text_height_x + frame_width*i, text_height_x))
        #original_gif.append(np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3))
        original_gif.append(img)

    # Prediction
    predicted_gif = []
    predicted_images_np = []
    for i in xrange(len(resize_predicted_images)):
        #img = resize_predicted_images[i].data[0]
        img = resize_predicted_images[i]
        img = np.rollaxis(img, 0, 3)
        img = Image.fromarray(cupy.asnumpy(img), 'RGB')

        # Resize the image to the original dimensions
        img = img.resize((original_image_width, original_image_height), Image.ANTIALIAS)

        if downscale_factor != 1:
            img = img.resize((frame_width, frame_height), Image.ANTIALIAS)

        new_image.paste(img, (text_height_x + frame_width*i, frame_height + text_height_x))
        #predicted_gif.append(np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3))
        predicted_gif.append(img)

    if gif == 1:
        pred_image.save(path + '/prediction-' + str(time_step) + '-' +  model_name + '_prediction.gif', save_all=True, append_images=predicted_gif, transparency=0)
        orig_image.save(path + '/prediction-' + str(time_step) + '-' +  model_name + '_original.gif', save_all=True, append_images=original_gif, transparency=0)

    np.save(('/home/user/Robotics/CDNA/CDNA/slip_detection/models/' + str(model_dir) + '/original_images'), np.array(img_pred))
    np.save(('/home/user/Robotics/CDNA/CDNA/slip_detection/models/' + str(model_dir) + '/predicted_images'), np.array(resize_predicted_images))

    print("saved gt and predicted  images")

    # If enabled, create a GIF from the sequence of original and predicted image
    if gif == 1:
        # Create a tmp file
        temp_original_gif_path = path + '/original-' + str(time_step) + model_name + '.gif'
        temp_predicted_gif_path = path + '/predicted-' + str(time_step) + model_name + '.gif'
        #imageio.mimsave(temp_original_gif_path, original_gif)
        #imageio.mimsave(temp_predicted_gif_path, predicted_gif)
        #original_gif_img = Image.open(temp_original_gif_path)
        #predicted_gif_img = Image.open(temp_predicted_gif_path)
        # Import the tmp file and reshape each frame to the whole scene width/height
        gif_frames = []
        for img in original_gif:
            reshaped_original_gif_img = Image.new('RGB', (total_width, total_height))
            reshaped_original_gif_img.paste(img, (text_height_x + frame_width * time_step, text_height_x))
            gif_frames.append(reshaped_original_gif_img)
        for img in predicted_gif:
            reshaped_predicted_gif_img = Image.new('RGB', (total_width, total_height))
            reshaped_predicted_gif_img.paste(img, (text_height_x + frame_width * time_step, text_height_x + frame_height))
            gif_frames.append(reshaped_predicted_gif_img)

        # Avoid flickering when gif is done: add a still under the gif
        new_image.paste(original_gif[0], (text_height_x + frame_width * time_step, text_height_x))
        new_image.paste(predicted_gif[0], (text_height_x + frame_width * time_step, text_height_x + frame_height))
        # Clean the tmp files
        #os.remove(temp_original_gif_path)
        #os.remove(temp_predicted_gif_path)

    if gif == 1:
        new_image.save(path + '/prediction-' + str(time_step) + '-' +  model_name + '.gif', save_all=True, append_images=gif_frames, transparency=0)
    else:
        new_image.save(path + '/prediction-' + str(time_step) + '-' +  model_name + '.png')

    # print(model)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
