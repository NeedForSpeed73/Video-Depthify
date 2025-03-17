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

	args = parser.parse_args()

	input_rgb_dir = args.input_rgb_dir
	output_dir = args.output_dir
	small_model = args.smallmodel

	if small_model:
		midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
	else:
		midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)	

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

		for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):

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

			#output_normalized = (output * 255 / np.max(output)).astype('uint8')
			#output_image = Image.fromarray(output_normalized)
			#output_image_converted = output_image.convert('L').save(file.replace('rgb', 'depth'))
			rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
			pred_name_base = rgb_name_base + "_pred"
			png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
			if os.path.exists(png_save_path):
				logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
			output_normalized = (output * 255 / np.max(output)).astype('uint8')
			output_image = Image.fromarray(output_normalized)
			output_image_converted = output_image.convert('L').save(png_save_path)
		print('Done.')
