import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def create_siamese_network(embedding_size):
    # Define the tensors for the two input image embeddings
    input_a = Input(shape=(embedding_size,))
    input_b = Input(shape=(embedding_size,))

    # Neural network to learn the embeddings
    shared_network = Dense(512, activation='relu', kernel_regularizer=l2(1e-3))
    shared_network = Dropout(0.5)
    shared_network = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))
    shared_network = Dropout(0.5)
    shared_network = Dense(128, activation='relu', kernel_regularizer=l2(1e-3))
    shared_network = Dropout(0.5)
    shared_network = Dense(64, activation='relu', kernel_regularizer=l2(1e-3))

    processed_a = shared_network(input_a)
    processed_b = shared_network(input_b)

    # Compute the Euclidean distance between the two embeddings
    distance = Lambda(euclidean_distance)([processed_a, processed_b])

    # Define the model
    model = Model(inputs=[input_a, input_b], outputs=distance)

    return model


def contrastive_loss(y_true, y_pred):
    margin = 1  # You can experiment with the margin value
    y_true = K.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
