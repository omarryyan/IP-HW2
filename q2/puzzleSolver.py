# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil
import sys

# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    src_points = matches[:, 0]  # Source points
    dst_points = matches[:, 1]  # Destination points

    if is_affine:
        # Ensure we have exactly 3 point pairs for affine transformation
        transform = cv2.estimateAffine2D(src_points, dst_points)
    else:
        transform, _ = cv2.findHomography(src_points.astype(np.float32), dst_points.astype(np.float32))

    return transform

def stitch(img1, img2):
    # Add your code here
    return None

# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    output_width, output_height = output_size

    if original_transform.shape == (3, 3):  # Augmented affine or projective transformation
        # Check if it's an affine transformation (last row [0, 0, 1])
        if np.allclose(original_transform[2], [0, 0, 1]):
            # Extract the top 2 rows for affine transformation
            affine_transform = original_transform[:2, :]
            transformed_img = cv2.warpAffine(target_img, affine_transform, (output_width, output_height))
        else:
            # Use the full 3x3 matrix for projective transformation
            transformed_img = cv2.warpPerspective(target_img, original_transform, (output_width, output_height))
    elif original_transform.shape == (2, 3):  # Regular affine transformation
        transformed_img = cv2.warpAffine(target_img, original_transform, (output_width, output_height))
    else:
        raise ValueError("Invalid transformation matrix shape.")

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

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1, affine, 2, 2)

    return matches, affine == 3, n_images

if __name__ == '__main__':
    lst = ['puzzle_affine_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        path1 = os.path.join(pieces_pth, 'piece_1.jpg')  # Target image (image1)
        path2 = os.path.join(pieces_pth, 'piece_2.jpg')  # Source image (image2)

        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        print(matches)
        print("-----------------------")
        print(is_affine)
        print("-----------------------")
        print(n_images)
        print("-----------------------")

        for idx in range(1, n_images):
            piece = cv2.imread(os.path.join(pieces_pth, f'piece_{idx + 1}.jpg'))

            # Compute transformation from source (image2) to target (image1)
            transform_tuple = get_transform(matches=matches[idx - 1], is_affine=is_affine)
            transform = transform_tuple[0]  # Extract the transformation matrix
            if transform is None:
                raise ValueError("Failed to compute transformation matrix.")

            transform = transform.astype(np.float32)  # Now safe to convert to float32

            # Compute the inverse transformation matrix
            if transform.shape == (2, 3):  # Affine transformation
                inverse_transform = transform  # Directly use the 2x3 matrix for affine transformations
            else:  # Homography transformation
                inverse_transform = np.linalg.inv(transform)

            # Apply the inverse transformation to image2
            aligned_image2 = inverse_transform_target_image(image2, inverse_transform, image1.shape[::-1])

            # Display the resulting image
            cv2.imshow('Aligned Image2', aligned_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save the solution (optional)
            sol_file = f'solution_piece_{idx + 1}.jpg'
            cv2.imwrite(os.path.join(puzzle, sol_file), aligned_image2)
