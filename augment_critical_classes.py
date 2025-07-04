import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# Augment only these classes
target_classes = ['glioma', 'pituitary']
input_path = 'tumordataset/Training'
augment_count = 300  # number of images to generate per class

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for class_name in target_classes:
    class_dir = os.path.join(input_path, class_name)
    images = os.listdir(class_dir)

    count = 0
    for img_name in images:
        if count >= augment_count:
            break
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        prefix = os.path.splitext(img_name)[0]
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=class_dir,
                                  save_prefix=f"{prefix}_aug",
                                  save_format='jpeg'):
            i += 1
            count += 1
            if i >= 2 or count >= augment_count:  # save 2 augmentations per image max
                break

    print(f"Generated {count} augmented images for {class_name}")
