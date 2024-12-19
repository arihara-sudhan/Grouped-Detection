import os
import random
from PIL import Image, ImageEnhance, ImageFilter

def augment_image(img):
    max_shift = 10
    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-max_shift, max_shift)
    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, x_shift, 0, 1, y_shift),
        resample=Image.BICUBIC
    )

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.0))

    return img

def augment_images_in_subfolders(root_folder, target_count=40):
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            images = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
            
            if not images:
                continue
            
            current_count = len(images)
            if current_count >= target_count:
                continue
            
            augment_count = target_count - current_count
            
            for i in range(augment_count):
                img_path = random.choice(images)
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    augmented_img = augment_image(img)
                    output_path = os.path.join(subfolder_path, f"AUG_{i+1}.jpg")
                    augmented_img.save(output_path)

root_folder = "./cropped"
augment_images_in_subfolders(root_folder)
