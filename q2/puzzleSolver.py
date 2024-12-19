# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil
import sys

#matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
	src_points = matches[:, 0]  # Source points (e.g., in the canvas)
	dst_points = matches[:, 1]  # Destination points (e.g., in the transformed piece)
	print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=")
	print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=")
	print(src_points)
	print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=")
	print(dst_points)
	print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=")
	print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=")
	if is_affine:
		# Ensure we have exactly 3 point pairs for affine transformation
		if len(src_points) != 3 or len(dst_points) != 3:
			raise ValueError("Affine transformation requires exactly 3 point pairs.")
		transform = cv2.getAffineTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
	else:
		# Ensure we have at least 4 point pairs for projective transformation
		if len(src_points) < 4 or len(dst_points) < 4:
			raise ValueError("Projective transformation requires at least 4 point pairs.")
		transform, _ = cv2.findHomography(src_points.astype(np.float32), dst_points.astype(np.float32))

	return transform

def stitch(img1, img2):
	# Add your code here
	return None

# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
	# Add your code here

	output_width, output_height = output_size
	if original_transform.shape == (2, 3):  # Affine transformation
		transformed_img = cv2.warpAffine(target_img, original_transform, (output_width, output_height))
	else:  # Projective transformation
		transformed_img = cv2.warpPerspective(target_img, original_transform, (output_width, output_height))

	return transformed_img

# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
	edited = os.path.join(puzzle_dir, 'abs_pieces')
	if os.path.exists(edited):
		shutil.rmtree(edited)
	os.mkdir(edited)
	
	affine = 4 - int("affine" in puzzle_dir)
	
	matches_data = os.path.join(puzzle_dir, 'matches.txt')
	n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

	matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1,affine,2,2)
	
	return matches, affine == 3, n_images


if __name__ == '__main__':
	#lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
	lst = ['puzzle_affine_1']

	for puzzle_dir in lst:
		print(f'Starting {puzzle_dir}')
		
		puzzle = os.path.join('puzzles', puzzle_dir)
		
		pieces_pth = os.path.join(puzzle, 'pieces')
		edited = os.path.join(puzzle, 'abs_pieces')
		
		matches, is_affine, n_images = prepare_puzzle(puzzle)

		print(matches)
		print("-----------------------")
		print(is_affine)
		print("-----------------------")
		print(n_images)
		print("-----------------------")

		transform = get_transform(matches[0], is_affine)
		print(transform)
		sol_file = f'solution.jpg'
		#cv2.imwrite(os.path.join(puzzle, sol_file), )
