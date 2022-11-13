import cv2
import numpy as np

class HOG:

    def __call__(self, img_path):
        img = self.read_image(img_path)
        Ix, Iy = self.calculate_gradients(img)
        G, theta = self.calculate_magnitude_and_direction(Ix, Iy)
        G_max, theta_max = self.find_maximum_grad_along_color_channels(G, theta)
        feature_descriptor = self.compute_feature_descriptor(G_max, theta_max)
        return feature_descriptor

    def read_image(self, img_path):
        # convert image to size 64x128
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 128))
        return img

    def calculate_gradients(self, img):
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
        return Ix, Iy

    def calculate_magnitude_and_direction(self, Ix, Iy):
        G, theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
        return G, theta

    def find_maximum_grad_along_color_channels(self, G, theta):
        idx = np.argmax(G, axis=2)
        G_max = np.zeros((128, 64))
        theta_max = np.zeros((128, 64))
        for i in range(128):
            for j in range(64):
                G_max[i, j] = G[i, j, idx[i, j]]
                theta_max[i, j] = theta[i, j, idx[i, j]]
        theta_max[theta_max >= 180] -= 180
        return G_max, theta_max

    def compute_feature_descriptor(self, G_max, theta_max):
        feature_descriptor = np.array([])
        # sweep 16 x 16 window one pixel at a time across image
        for i in range(15):
            for j in range(7):
                G_window = G_max[i:i + 16, j:j + 16]
                theta_window = theta_max[i:i + 16, j:j + 16]
                window_descriptor = np.array([])
                for k in range(2):
                    for l in range(2):
                        block_descriptor = np.zeros((9,))
                        G_block = G_window[k * 8:(k + 1) * 8, l * 8:(l + 1) * 8].reshape(-1)
                        theta_block = theta_window[k * 8:(k + 1) * 8, l * 8:(l + 1) * 8].reshape(-1)
                        for m in range(len(G_block)):
                            x = int(np.floor(theta_block[m] / 20)) % 9
                            y = (x + 1) % 9
                            block_descriptor[x] += abs(theta_block[m] - x * 20) / 20
                            block_descriptor[y] += abs(theta_block[m] - y * 20) / 20
                        window_descriptor = np.concatenate([window_descriptor, block_descriptor])
                window_descriptor = window_descriptor / np.linalg.norm(window_descriptor)
                feature_descriptor = np.concatenate([feature_descriptor, window_descriptor])
        return feature_descriptor

hog = HOG()
hog_feature_descriptor = hog('puppy.png')