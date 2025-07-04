import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = load_model('model/multiclass_vgg19_model.h5')

# Define class names
# Match folder names exactly
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class

def evaluate_on_test_folder():
    test_dir = 'tumordataset/Testing'
    y_true = []
    y_pred = []

    for i, class_name in enumerate(class_names):
        class_folder = os.path.join(test_dir, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            prediction = model_predict(img_path)
            y_pred.append(prediction)
            y_true.append(class_name)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Print accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# Only run evaluation when executing this file directly
if __name__ == '__main__':
    evaluate_on_test_folder()
