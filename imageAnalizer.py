import cv2

class ImageAnalizer:
    def __init__(self):
        pass

    def get_grayscale_array(self, image_name, width, height):
        # Open image
        img = cv2.imread(image_name)
        # Switch to grayscale
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray_scale_array = []
        for row in range(height):
            for column in range(width):
                gray_scale_array.append(img2[row, column]/1000)

        print(gray_scale_array)
        return gray_scale_array
