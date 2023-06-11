# import các thư viện
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array
from tensorflow.keras.utils import load_img
import numpy as np
import os

# Truyền đường dẫn folder chứa ảnh đầu vào và đường dẫn folder chứa ảnh đầu ra
input_folder = 'animals/cat'
output_folder = 'output2/cat'
prefix = 'image'

# Lấy danh sách các file ảnh trong folder
input_image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# Khởi tạo bộ sinh ảnh tăng cường và khởi tạo tổng số ảnh được sinh ra
aug = ImageDataGenerator(

                        rescale=1. / 255,
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')
total = 0

# Lặp qua từng file ảnh trong folder và thực hiện tăng cường
for input_path in input_image_paths:
    # Nạp ảnh đầu vào, convert it sang mảng NumPy array, rồi reshape nó
    print("[INFO] Nạp image: ", input_path)
    image = load_img(input_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Sinh ảnh tăng cường và lưu vào thư mục đầu ra
    print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=output_folder, save_prefix=prefix, save_format="jpg")

    # Lặp qua các ảnh đã được tăng cường ảnh trong imageGen
    for image in imageGen:
        # Tăng bộ đếm
        total += 1
        # Lặp cho đến khi sinh ra đủ số ảnh cần thiết (ở đây là 10 ảnh cho mỗi file ảnh đầu vào)
        if total == 20:
            break
    # Đặt lại bộ đếm về 0 cho file ảnh tiếp theo
    total = 0
