# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil

# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    src_points = matches[:, 0].astype(np.float32)  # Source points
    dst_points = matches[:, 1].astype(np.float32)  # Destination points

    if is_affine:
        # Ensure we have exactly 3 point pairs for affine transformation
        transform = cv2.getAffineTransform(src_points, dst_points)
        # Convert to 3x3 by appending [0, 0, 1] for consistency
        transform = np.vstack([transform, [0, 0, 1]])
    else:
        # Perspective transformation (Homography)
        transform = cv2.getPerspectiveTransform(src_points, dst_points)

    return transform

# stitch img2 to img1
def stitch(img1, img2):
    # Create a mask where img2 has non-black pixels
    mask = np.any(img2 != 0, axis=-1).astype(np.uint8)

    # Overlay img2 onto img1 using the mask
    cv2.copyTo(img2, mask, img1)

    return img1

# Function returns the original image of target_img using transform matrix. Output image size is output_size. Output size is (w,h)
def inverse_transform_target_image(target_img, transform, output_size):
    output_width, output_height = output_size

    # Check if it's an affine transformation (last row [0, 0, 1])
    if np.allclose(transform[2], [0, 0, 1]):
        # Extract the top 2 rows for affine transformation
        affine_transform = transform[:2, :]
        transformed_img = cv2.warpPerspective(target_img, transform, (output_width, output_height), flags=2, borderMode=cv2.BORDER_TRANSPARENT)
    else:
        # Use the full 3x3 matrix for projective transformation
        transformed_img = cv2.warpPerspective(target_img, transform, (output_width, output_height), flags=2, borderMode=cv2.BORDER_TRANSPARENT)
    return transformed_img

# returns list of pieces file names
def prepare_puzzle(puzzle_dir):

    edited = os.path.join(puzzle_dir, 'abs_pieces')

    # Clear out the previous edited directory if it exists
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_2']  # List of puzzles to process

    # Loop through each puzzle directory
    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        # Directly load image files (either .jpg or .png) from the puzzle pieces directory
        filenames = sorted([file for file in os.listdir(pieces_pth) if file.endswith((".jpg", ".png"))])
        images = [cv2.imread(os.path.join(pieces_pth, filename)) for filename in filenames]

        image1 = images[0]  # The first image is treated as the target image (image1)

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        solution = image1  # Initialize solution with the first image (target)

        # Loop through all puzzle pieces starting from index 1
        for idx in range(1, n_images):
            piece = images[idx]

            # Compute the transformation from the current piece to the target image (image1)
            transform = get_transform(matches=matches[idx - 1], is_affine=is_affine)

            # Convert the transformation matrix to float32 type
            transform = transform.astype(np.float32)

            # Compute the inverse of the transformation matrix
            inverse_transform = np.linalg.inv(transform)

            # Apply the inverse transformation to align the piece with the target image
            output_height, output_width = image1.shape[:2]
            aligned_image = inverse_transform_target_image(piece, inverse_transform, (output_width, output_height))

            # Display the aligned image
            cv2.imshow(f'Aligned Image{idx + 1}', aligned_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Stitch the aligned image onto the solution (progressive stitching)
            solution = stitch(solution, aligned_image)
            cv2.imshow(f'Solution{idx + 1}', solution)
            cv2.waitKey(0)

            # Optionally save the aligned image as a part of the solution
            sol_file = f'solution_piece_{idx + 1}.jpg'
            cv2.imwrite(os.path.join(edited, sol_file), aligned_image)
