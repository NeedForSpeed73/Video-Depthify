#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import urllib.request
from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse
import logging

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def process_rgb_to_depth(midas, device, transform, rgb_path):
    img = cv2.imread(rgb_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1),
				size = img.shape[:2],
				mode = 'bicubic',
				align_corners = False,
			).squeeze()

    output = prediction.cpu().numpy()
    return output

def save_normalized_depth_image(output_dir_tif, path_name, image):
    rgb_name_base = os.path.splitext(os.path.basename(path_name))[0]
    pred_name_base = rgb_name_base + "_pred"
    png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
    if os.path.exists(png_save_path):
     logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
    image_normalized = (image * 255 / np.max(image)).astype('uint8')
    image = Image.fromarray(image_normalized)
    image_converted = image.convert('L').save(png_save_path)

def average_neighborhood_values(arr, prev, curr, next):
    arr = arr + prev / 3
    arr = arr + curr / 3
    arr = arr + next / 3
    return arr

if "__main__" == __name__:
	logging.basicConfig(level=logging.INFO)

	parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using DPT_Large or MiDaS_small."
    	)

	parser.add_argument(
	"--input_rgb_dir",
	type=str,
	required=True,
	help="Path to the input image folder.",
	)

	parser.add_argument(
	"--output_dir",
	type=str,
	required=True,
	help="Output directory."
	)

	parser.add_argument(
	"--smallmodel",
	action="store_true",
	help="Use MiDaS small model.",
	)

	parser.add_argument(
	"--apple_silicon",
    action="store_true",
    help="Flag of running on Apple Silicon.",
	)

	args = parser.parse_args()
	
	input_rgb_dir = args.input_rgb_dir
	output_dir = args.output_dir
	small_model = args.smallmodel
	apple_silicon = args.apple_silicon

	if small_model:
		midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
	else:
		midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)

	if apple_silicon:
		if torch.backends.mps.is_available() and torch.backends.mps.is_built():
			device = torch.device("mps:0")
		else:
			device = torch.device("cpu")
			logging.warning("MPS is not available. Running on CPU will be slow.")
	else:
		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
			logging.warning("CUDA is not available. Running on CPU will be slow.")
	logging.info(f"device = {device}")

	midas.to(device)
	midas.eval()

	midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

	if small_model:
		transform = midas_transforms.small_transform
		print('Using small (fast) model.')
	else:
		transform = midas_transforms.dpt_transform
		print('Using large (slow) model.')

	output_dir_tif = os.path.join(output_dir, "depth_bw")
	os.makedirs(output_dir_tif, exist_ok=True)
	logging.info(f"output dir = {output_dir}")

	rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
	rgb_filename_list = [
		f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
	]
	rgb_filename_list = sorted(rgb_filename_list)
	n_images = len(rgb_filename_list)
	if n_images > 0:
		logging.info(f"Found {n_images} images")
	else:
		logging.error(f"No image found in '{input_rgb_dir}'")
		exit(1)

	with torch.no_grad():
		os.makedirs(output_dir, exist_ok=True)

		items = n_images - 2
		first = rgb_filename_list[0]
		last = rgb_filename_list[n_images - 1]

		# Process the first image
		# and save it to the output directory
		# with the same name as the input image
		# but with the "_pred" suffix
		
		output = process_rgb_to_depth(midas, device, transform, first)
		save_normalized_depth_image(output_dir_tif, first, output)

		# Process the second image with averaging
		# and save it to the output directory
		arr = np.zeros((output.shape[0], output.shape[1]), np.float64)
		prev = output
		curr = process_rgb_to_depth(midas, device, transform, rgb_filename_list[1])
		next = process_rgb_to_depth(midas, device, transform, rgb_filename_list[2])

		arr = average_neighborhood_values(arr, prev, curr, next)

		save_normalized_depth_image(output_dir_tif, rgb_filename_list[1], arr)
		
		# Process all other images with averaging
		# and save them to the output directory
		 
		current	= 2

		for idx in tqdm(range(items - 1)):
			current = idx + 2
			arr = np.zeros((output.shape[0], output.shape[1]), np.float64)
			prev = curr
			curr = next
			next = process_rgb_to_depth(midas, device, transform, rgb_filename_list[current + 1])

			arr = average_neighborhood_values(arr, prev, curr, next)
						
			save_normalized_depth_image(output_dir_tif, rgb_filename_list[current], arr)
	
		# Process the last image
		# and save it to the output directory
			
		output = process_rgb_to_depth(midas, device, transform, last)
		save_normalized_depth_image(output_dir_tif, last, output)
			
	print('Done.')
		
