# visualization.py

import matplotlib.pyplot as plt
import numpy as np

def test_model(model, X_val, Y_val, index=0):
    grayscale_img = X_val[index]
    true_color_img = Y_val[index]
    grayscale_img_batch = np.expand_dims(grayscale_img, axis=0)

    predicted_color_img = model.predict(grayscale_img_batch)[0]

    predicted_color_img = (predicted_color_img * 255).astype(np.uint8)
    true_color_img = (true_color_img * 255).astype(np.uint8)
    grayscale_img = (grayscale_img.squeeze() * 255).astype(np.uint8)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Grayscale Input")
    plt.imshow(grayscale_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Color Image")
    plt.imshow(predicted_color_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("True Color Image")
    plt.imshow(true_color_img)
    plt.axis('off')

    plt.show()
