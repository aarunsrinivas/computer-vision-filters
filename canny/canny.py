import cv2
import numpy as np
import matplotlib.pyplot as plt

class CannyEdgeDetector:

    def __init__(self, weak_threshold=None, strong_threshold=None):
        self.weak_threshold = weak_threshold
        self.strong_threshold = strong_threshold

    def __call__(self, img_path):
        img = self.read_image(img_path)
        img = self.reduce_noise(img)
        Ix, Iy = self.calculate_gradients(img)
        G, theta = self.calculate_magnitude_and_direction(Ix, Iy)
        G_suppress = self.non_max_suppression(img, G, theta)
        G_threshold = self.double_thresholding(G_suppress)
        return G_threshold

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def reduce_noise(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 1.4)
        return img

    def calculate_gradients(self, img):
        Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
        Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
        return Ix, Iy

    def calculate_magnitude_and_direction(self, Ix, Iy):
        G, theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
        theta[theta > 180] -= 180 # remove opposite facing vectors
        return G, theta

    def non_max_suppression(self, img, G, theta):
        G = G.copy()
        M, N = img.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                # find neighbors along gradient angle for each pixel
                n1, n2 = None, None
                # covers x-axis
                if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                    n1, n2 = G[i, j - 1], G[i, j + 1]
                # covers Q1-Q3 diagonal
                elif 22.5 <= theta[i, j] < 67.5:
                    n1, n2 = G[i + 1, j - 1], G[i - 1, j + 1]
                # covers y-axis
                elif 67.5 <= theta[i, j] < 112.5:
                    n1, n2 = G[i + 1, j], G[i - 1, j]
                # covers Q2-Q4 diagonal
                elif 112.5 <= theta[i, j] < 157.5:
                    n1, n2 = G[i + 1, j + 1], G[i - 1, j - 1]
                # non-max suppression step
                if G[i, j] < n1 or G[i, j] < n2:
                    G[i, j] = 0
        return G

    def double_thresholding(self, G):
        G = G.copy()
        G_max = np.max(G)
        weak_threshold = self.weak_threshold or G_max * 0.1
        # strong_threshold = self.strong_threshold or G_max * 0.5
        G[G < weak_threshold] = 0
        return G

canny_edge_detector = CannyEdgeDetector()
canny_img = canny_edge_detector('mountain.png')
plt.imshow(canny_img)
plt.show()