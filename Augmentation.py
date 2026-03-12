


import os
import numpy as np
from PIL import Image

# Original dataset
image_dir = r'D:\segmentation\Data\train\image'
mask_dir = r'D:\segmentation\Data\train\mask'

# Augmented dataset (separate folder)
aug_image_dir = r'D:\segmentation\Data\train_augmented\image'
aug_mask_dir = r'D:\segmentation\Data\train_augmented\mask'

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Number of augmented samples per image
augmentation_count = 1

for filename in os.listdir(image_dir):

    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if not os.path.exists(mask_path):
        continue

    # Load image and mask
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))

    for i in range(augmentation_count):

        # Horizontal flip
        aug_image = np.fliplr(image)
        aug_mask = np.fliplr(mask)

        new_name = filename.replace(".png", f"_aug_{i}.png")

        Image.fromarray(aug_image).save(os.path.join(aug_image_dir, new_name))
        Image.fromarray(aug_mask).save(os.path.join(aug_mask_dir, new_name))

print("Augmentation completed successfully!")





# vertical flip





import os
import numpy as np
from PIL import Image

# Original dataset
image_dir = r'D:\segmentation\Data\train\image'
mask_dir = r'D:\segmentation\Data\train\mask'

# Augmented dataset (separate folder)
aug_image_dir = r'D:\segmentation\Data\train_augmented_vertical\image'
aug_mask_dir = r'D:\segmentation\Data\train_augmented_vertical\mask'

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Number of augmented samples per image
augmentation_count = 1

for filename in os.listdir(image_dir):

    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if not os.path.exists(mask_path):
        continue

    # Load image and mask
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))

    for i in range(augmentation_count):

        # Vertical flip
        aug_image = np.flipud(image)
        aug_mask = np.flipud(mask)

        new_name = filename.replace(".png", f"_vflip_{i}.png")

        Image.fromarray(aug_image).save(os.path.join(aug_image_dir, new_name))
        Image.fromarray(aug_mask).save(os.path.join(aug_mask_dir, new_name))

print("Vertical flip augmentation completed successfully!")





#random Roatation 





import os
import numpy as np
import random
from PIL import Image

# Original dataset
image_dir = r'F:\segmentation\Data\train\image'
mask_dir = r'F:\segmentation\Data\train\mask'

# Augmented dataset (separate folder)
aug_image_dir = r'F:\segmentation\Data\train_augmented_rotation\image'
aug_mask_dir = r'F:\segmentation\Data\train_augmented_rotation\mask'

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Number of augmented samples per image
augmentation_count = 1

for filename in os.listdir(image_dir):

    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if not os.path.exists(mask_path):
        continue

    # Load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    for i in range(augmentation_count):

        # Random angle between -30 and +30
        angle = random.uniform(-30, 30)

        # Rotate image and mask with same angle
        aug_image = image.rotate(angle)
        aug_mask = mask.rotate(angle)

        new_name = filename.replace(".png", f"_rot_{i}.png")

        aug_image.save(os.path.join(aug_image_dir, new_name))
        aug_mask.save(os.path.join(aug_mask_dir, new_name))

print("Random rotation augmentation completed successfully!")





#Shear Transformation





import os
import numpy as np
import random
from PIL import Image

# Original dataset
image_dir = r'D:\segmentation\Data\train\image'
mask_dir = r'D:\segmentation\Data\train\mask'

# Augmented dataset (separate folder)
aug_image_dir = r'D:\segmentation\Data\train_augmented_shear\image'
aug_mask_dir = r'D:\segmentation\Data\train_augmented_shear\mask'

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Number of augmented samples per image
augmentation_count = 1

for filename in os.listdir(image_dir):

    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if not os.path.exists(mask_path):
        continue

    # Load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    width, height = image.size

    for i in range(augmentation_count):

        # Random shear angle between -10 and +10 degrees
        shear_angle = random.uniform(-10, 10)

        # Convert angle to shear factor
        shear_factor = np.tan(np.radians(shear_angle))

        # Affine transformation matrix
        matrix = (1, shear_factor, 0,
                  0, 1, 0)

        aug_image = image.transform(
            (width, height),
            Image.AFFINE,
            matrix,
            resample=Image.BILINEAR
        )

        aug_mask = mask.transform(
            (width, height),
            Image.AFFINE,
            matrix,
            resample=Image.NEAREST
        )

        new_name = filename.replace(".png", f"_shear_{i}.png")

        aug_image.save(os.path.join(aug_image_dir, new_name))
        aug_mask.save(os.path.join(aug_mask_dir, new_name))

print("Shear augmentation completed successfully!")





