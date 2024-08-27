import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

class ImageProcessor:
    def __init__(self, image_path=None, image_array=None):
        if image_path is not None:
            self.image_path = image_path
            self.image = self._load_image_from_path()
        elif image_array is not None:
            self.image = self._load_image_from_array(image_array)
        else:
            raise ValueError("Either image_path or image_array must be provided.")
        self.whitebalanced_image = None
        self.quantized_image = None
        self.color_group = None

    def _load_image_from_path(self):
        # 이미지 로드 (OpenCV는 BGR로 이미지를 로드하므로 RGB로 변환합니다)
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to floats instead of the default 8 bits integer coding.
        image = np.array(image, dtype=np.float64) / 255
        return image
    
    def _load_image_from_array(self, image_array):
        # 이미지가 이미 RGB 형식인지 확인
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Input array must be an RGB image with shape (height, width, 3).")
        image = np.array(image_array, dtype=np.float64) / 255
        return image
    
    def apply_whitebalance(self, percentile_value=95):
        self.whitebalanced_image = img_as_ubyte(
            (self.image * 1.0 / np.percentile(self.image, percentile_value, axis=(0, 1))).clip(0, 1)
        )
        return self.whitebalanced_image

    def apply_kmeans_quantization(self, n_colors=5, background_threshold=244):
        if self.whitebalanced_image is None:
            raise ValueError("Please apply whitebalance before quantization.")
        
        # Convert image to uint8 and [0, 255] range if needed
        image = self.whitebalanced_image
        if image.dtype == np.float64 and np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create a mask to exclude the background
        mask = np.all(image > background_threshold, axis=-1)
        masked_image = image[~mask]

        # Reshape the masked image to a 2D array of pixels
        w, h, d = original_shape = tuple(image.shape)
        image_array = masked_image.reshape(-1, d)

        # Fit the KMeans model on the masked image data
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1000)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        # Predict color indices for the masked pixels
        labels = kmeans.predict(image_array)

        # Recreate the image
        quantized_image = np.full_like(image, fill_value=255)  # Start with a white image
        quantized_image[~mask] = kmeans.cluster_centers_[labels].astype(np.uint8)
        self.quantized_image = quantized_image

        # Calculate color percentages and counts
        label_counts = np.bincount(labels)
        percentages = np.around(((label_counts / len(labels)) * 100), 2)
        colors = []
        for i, (percentage, count) in enumerate(zip(percentages, label_counts)):
            color = kmeans.cluster_centers_[i].astype(int)
            colors.append(rgb_to_hex(color.tolist()))
        
        self.color_group = sorted(zip(colors, percentages), key=lambda x: x[1], reverse=True)
        return self.quantized_image, self.color_group

    def process_image(self, percentile_value=95, n_colors=5, background_threshold=244):
        return self.apply_whitebalance(percentile_value), self.apply_kmeans_quantization(n_colors, background_threshold)

