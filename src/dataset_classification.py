import os
import numpy as np
from PIL import Image

from utils.data import CSVHandler, ImageHandler


def filter_dataset(path_csv):
    if not os.path.exists(path_csv):
        raise FileNotFoundError(path_csv)

    # create csv handler con path al csv completo
    csv = CSVHandler(path_csv)
    # prendere tutti i filename
    # TODO: check possible error on multiple entries in filename array
    for filename in csv.get_images_name():
        print(filename)
        # iterare sui filename
        filename_path = os.path.join(os.path.dirname(path_csv), '..', 'images', filename)

        # check if image exist for 'filename' entry in csv
        if os.path.exists(filename_path):
            # if true checks all the entries for 'filename' image
            img = ImageHandler(filename)
            data = csv.get_image_data(filename)

            for i, row in enumerate(data):
                # loop over the various desired crop sizes and save in different datasets for each size
                for size in [64, 128, 256, 512]:
                    # tensor of the cropped image
                    crop_img = img.img_crop_and_resize(row[4], row[5], row[6], row[7],
                                                       [size, size])
                    # numpy array of the cropped image
                    img_array = crop_img.numpy().astype(np.uint8)
                    # pillow image
                    pil_img = Image.fromarray(img_array)

                    dataset_path = os.path.join(os.path.dirname(path_csv),
                                                '..',
                                                '..',
                                                'dataset'+str(size),
                                                str(row[3]))
                    # checks if path to the correct dataset folder exists
                    if not os.path.exists(dataset_path):
                        os.makedirs(dataset_path)

                    # creates path to image
                    img_path = os.path.join(dataset_path,
                                            filename.replace(".jpg", "_")
                                            +"Obj"+str(i)+"_"
                                            +row[3]+".jpg")

                    # saves image in correct path "datasetSIZE/CLASS/"
                    pil_img.save(img_path)

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    filter_dataset(os.path.join(path, '..', 'datasets',
                                'spqr_dataset', 'raw', 'bbx_annotations.csv'))
