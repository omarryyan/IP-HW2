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

def stitch(img1, img2):
    """
    Combine two images, overlaying non-black pixels from img2 onto img1.

    Parameters:
    - img1: The base image (numpy array).
    - img2: The overlay image (numpy array).

    Returns:
    - The combined image.
    """
    # Create a mask where img2 has non-black pixels
    mask = np.any(img2 != 0, axis=-1).astype(np.uint8)

    # Overlay img2 onto img1 using the mask
    #img1[mask]=img2[mask]
    cv2.copyTo(img2, mask, img1)

    return img1

# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    output_width, output_height = output_size
    print(output_width, output_height)

    if original_transform.shape == (3, 3):  # Augmented affine or projective transformation
        # Check if it's an affine transformation (last row [0, 0, 1])
        if np.allclose(original_transform[2], [0, 0, 1]):
            # Extract the top 2 rows for affine transformation
            affine_transform = original_transform[:2, :]
            transformed_img = cv2.warpPerspective(target_img, original_transform, (output_width, output_height), flags=cv2.INTER_CUBIC)
        else:
            # Use the full 3x3 matrix for projective transformation
            transformed_img = cv2.warpPerspective(target_img, original_transform, (output_width, output_height), flags=cv2.INTER_CUBIC)

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
    lst = ['puzzle_affine_2']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        path1 = os.path.join(pieces_pth, 'piece_1.jpg')  # Target image (image1)
        path2 = os.path.join(pieces_pth, 'piece_2.jpg')  # Source image (image2)

        image1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(path2, cv2.IMREAD_COLOR)

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        print(matches)
        print("-----------------------")
        print(is_affine)
        print("-----------------------")
        print(n_images)
        print("-----------------------")
        solution=image1

        for idx in range(1, n_images):
            piece = cv2.imread(os.path.join(pieces_pth, f'piece_{idx + 1}.jpg'))

            # Compute transformation from source (piece) to target (image1)
            transform = get_transform(matches=matches[idx - 1], is_affine=is_affine)

            transform = transform.astype(np.float32)  # Now safe to convert to float32

            inverse_transform = np.linalg.inv(transform)

            # Apply the inverse transformation to piece
            output_height, output_width = image1.shape[:2]
            aligned_image = inverse_transform_target_image(piece, inverse_transform, (output_width, output_height))

            # Display the resulting image
            cv2.imshow(f'Aligned Image{idx+ 1}', aligned_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            solution = stitch(solution, aligned_image)
            cv2.imshow(f'Solution{idx+1}', solution)
            cv2.waitKey(0)

            # Save the solution (optional)
            sol_file = f'solution_piece_{idx + 1}.jpg'
            cv2.imwrite(os.path.join(edited, sol_file), aligned_image)


