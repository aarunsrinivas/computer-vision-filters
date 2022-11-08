import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detector(img_path):
    # read image and convert to grayscale
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    # calculate gradients in the x and y direction
    Ix = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    Iy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # calculate magnitude and direction of gradients
    G, theta = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)

    # eliminates vectors that are opposite to vectors with theta 0-180
    theta[theta > 180] -= 180

    M, N = img.shape

    # non-max suppression algorithm
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


    # double thresholding
    G_max = np.max(G)
    weak_threshold = G_max * 0.1
    strong_threshold = G_max * 0.5
    G[G < weak_threshold] = 0
    # result = np.zeros_like(img)

    # strong_coordinates = np.where(Z >= strong_threshold)
    # weak_coordinates = np.where((Z >= weak_threshold) & (Z < strong_threshold))

    # result[strong_coordinates] = 255
    # result[weak_coordinates] = 25

    #TODO need to add hysteris step

    return G
    
canny_img = canny_edge_detector('mountain.png')
plt.imshow(canny_img)
plt.show()
