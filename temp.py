# import tensorflow as tf

# model_path = "insightface/models/model-y1-test2/model-0000"  # Update this path with the correct location of the SavedModel file

# hdf5_model_path = 'insightface/models/model-y1-test2/model.h5'  # Path to save the converted model

# try:
#     # Try to load the model directly
#     model = tf.keras.models.load_model(model_path)
#     print("Model loaded successfully.")
# except (OSError, ValueError):
#     # If loading directly fails, try converting the model to HDF5 format
#     print("Converting model to HDF5 format...")
#     converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
#     tflite_model = converter.convert()
    
#     # Save the model in HDF5 format
#     with open(hdf5_model_path, 'wb') as f:
#         f.write(tflite_model)
    
#     print("Model converted and saved to HDF5 format.")

#     # Load the converted model
#     model = tf.keras.models.load_model(hdf5_model_path)
#     print("Converted model loaded successfully.")

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load embeddings from embeddings.pkl
with open("artifacts/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
names = data["names"]

# Assuming you have a list of true embeddings for each image
# Replace true_embeddings with your actual embeddings
true_embeddings = [...]  

# Compare the loaded embeddings with the true embeddings
for i, (emb, true_emb) in enumerate(zip(embeddings, true_embeddings)):
    similarity = cosine_similarity([emb], [true_emb])[0][0]
    print(f"Similarity for {names[i]}: {similarity}")

# Visualize embeddings using PCA
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=range(len(names)))
plt.title('PCA Visualization of Face Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Image Index')
plt.show()
