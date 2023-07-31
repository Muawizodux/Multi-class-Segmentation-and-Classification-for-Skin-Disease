import cv2
import os
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge([cl, a, b])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result


def gamma_correction(image: np.ndarray, gamma):
    gamma_corrected = np.power(image/255.0, gamma)
    gamma_corrected = gamma_corrected*255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    return gamma_corrected


root_dir = "Queensland Dataset CE42/"
classes = ["BCC/", "IEC/", "SCC/"]
images_list = []
io_files = []

for idx, obj in enumerate(classes):
    image_file = os.listdir(root_dir + obj + "Images/")

    for _, image in enumerate(image_file):
        img = root_dir + obj + "Images/" + image
        images_list.append(img)


num_cores = multiprocessing.cpu_count()
print(f"Number of cores: {num_cores}" )


# Preprocessing the image with CLAHE and Gamma-correction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    clahe_img = apply_clahe(image)
    gamma_img = gamma_correction(clahe_img, 3)
    resize_img = cv2.resize(gamma_img, (256, 256))
    return resize_img


# Processing image in parallel using ThreadPoolExecutor
def process_image_parallel(image):
    return preprocess_image(image)


batch_size = 50
preprocessed_image = []
with ThreadPoolExecutor(max_workers=4) as executor:
    for i in range(0, len(images_list), batch_size):
        batch_images = images_list[i : i + batch_size]

        # Using ThreadPoolExecutor.map to to preprocess images in parallel
        preprocess_batch = list(executor.map(process_image_parallel, batch_images))
        preprocessed_image.extend(preprocess_batch)


for idx, obj in enumerate(classes):
    mask_files = os.listdir(root_dir + obj + "Masks/")
    for idx, mask in enumerate(mask_files):
      mask = cv2.imread(root_dir + obj + "Masks/" + mask)
      r_mask = cv2.resize(mask, (256, 256))

      io_files.append(cv2.hconcat([preprocessed_image[idx], r_mask]))


np.save("Preprocessed_data", np.array(io_files))
cv2.imshow("Image + mask: ", io_files[0])
cv2.waitKey(0)