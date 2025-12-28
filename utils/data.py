import os
import pandas as pd
import tensorflow.keras.preprocessing.image as preproc_image
import tensorflow.image as tf_image

from tensorflow import expand_dims
from PIL import Image



class ImageHandler:
    def __init__(self, file_name):
        self.image_name = file_name
        self.image = self.load_img(self.image_name)

    def load_img(self, file_name):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, "../datasets/spqr_dataset/images", file_name)
        image = Image.open(path)
        self.image = image
        return image

    def img_crop_and_resize(self, xmin, ymin, xmax, ymax, size=None):
        """
        This method crops the image and resize it to desired size.
        :param xmin: X coordinate of the top left corner
        :param ymin: Y coordinate of the top left corner
        :param xmax: X coordinate of the bottom right corner
        :param ymax: Y coordinate of the bottom right corner
        :param size: Desired size of the cropped image defined as [HxW]. If not given defaults to 64x64.
        :return: Cropped image
        """
        if size is None:
            size = [64, 64]

        # convert to 3D numpy array
        image_array = preproc_image.img_to_array(self.image)

        # normalize crop size
        norm_xmin = xmin / image_array.shape[1]
        norm_xmax = xmax / image_array.shape[1]

        norm_ymin = ymin / image_array.shape[0]
        norm_ymax = ymax / image_array.shape[0]

        # add dimension as new first axis
        image_array = expand_dims(image_array, axis=0)
        return tf_image.crop_and_resize(image_array,
                                 [[norm_ymin, norm_xmin, norm_ymax, norm_xmax]],
                                 [0],
                                 size,
                                 method='bilinear')

class CSVHandler:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def get_images_name(self):
        return self.df['filename'].to_numpy()

    def get_image_data(self, filename):
        return self.df[self.df['filename'] == filename].to_numpy()