import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift(img_path, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    base_img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
    
    num_octaves = int(round(np.log(np.min(base_img.shape)) / np.log(2) - 1))
    num_images_per_octave = num_intervals + 3

    x = 2 ** (1 / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for i in range(1, num_images_per_octave):
        sigma_prev = (x ** (i - 1)) * sigma
        sigma_total = x * sigma_prev
        gaussian_kernels[i] = np.sqrt(sigma_total ** 2 - sigma_prev ** 2)

    gaussian_images = []

    octave_base_img = base_img
    for i in range(num_octaves):
        gaussian_images_in_octave = [octave_base_img]
        for kernel in gaussian_kernels[1:]:
            octave_img = cv2.GaussianBlur(octave_base_img, (0, 0), sigmaX=kernel, sigmaY=kernel)
            gaussian_images_in_octave.append(octave_img)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base_img = gaussian_images_in_octave[-3]
        x, y = octave_base_img.shape
        octave_base_img = cv2.resize(octave_base_img, (y // 2, x // 2), interpolation=cv2.INTER_NEAREST)

    dog_images = []
    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(second_image - first_image)
        dog_images.append(dog_images_in_octave)

    def is_pixel_extrenum(first_subimage, second_subimage, third_subimage, threshold=0.04):
        center_pixel_value = abs(second_subimage[1, 1])
        if center_pixel_value > threshold:
            return np.all(center_pixel_value >= first_subimage) \
                and np.all(center_pixel_value >= second_subimage) \
                and np.all(center_pixel_value >= third_subimage)
        return False

    def compute_keypoints_with_orientations(keypoint, octave_idx, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        pass

    for octave_idx, dog_images_in_octave in enumerate(dog_images):
        for (first_image, second_image, third_image) \
            in zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:]):
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    first_subimage = first_image[i - 1: i + 2, j - 1: j + 2]
                    second_subimage = second_image[i - 1: i + 2, j - 1: j + 2]
                    third_subimage = third_image[i - 1: i + 2, j - 1: j + 2]
                    if is_pixel_extrenum(first_subimage, second_subimage, third_subimage):
                        pass

    


    




    # plt.imshow(img)
    # plt.show()
    


sift('box.png')