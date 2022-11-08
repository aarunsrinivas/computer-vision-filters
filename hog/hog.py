import cv2
import numpy as np

def hog_feature_descriptor(img_path):

    # convert image to size 64x128
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 128))

    # calculate gradients in the x and y direction
    Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # calculate magnitude and direction of gradients
    G, theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
    
    idx = np.argmax(G, axis=2)
    G_max = np.zeros((128, 64))
    theta_max = np.zeros((128, 64))
    for i in range(128):
        for j in range(64):
            G_max[i, j] = G[i, j, idx[i, j]]
            theta_max[i, j] = theta[i, j, idx[i, j]]

    theta_max[theta_max >= 180] -= 180

    image_vector = np.array([])
    for i in range(15):
        for j in range(7):
            G_window = G_max[i:i + 16, j:j + 16]
            theta_window = theta_max[i:i + 16, j:j + 16]
            window_vector = np.array([])
            for k in range(2):
                for l in range(2):
                    block_vector = np.zeros((9,))
                    G_block = G_window[k * 8:(k + 1) * 8, l * 8:(l + 1) * 8].reshape(-1)
                    theta_block = theta_window[k * 8:(k + 1) * 8, l * 8:(l + 1) * 8].reshape(-1)
                    for m in range(len(G_block)):
                        x = int(np.floor(theta_block[m] / 20)) % 9
                        y = (x + 1) % 9
                        block_vector[x] += abs(theta_block[m] - x * 20) / 20
                        block_vector[y] += abs(theta_block[m] - y * 20) / 20
                    window_vector = np.concatenate([window_vector, block_vector])
            window_vector = window_vector / np.linalg.norm(window_vector)
            image_vector = np.concatenate([image_vector, window_vector])
    
    return image_vector

hog_feature = hog_feature_descriptor('puppy.png')