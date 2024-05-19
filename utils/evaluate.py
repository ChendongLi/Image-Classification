# TODO evaluate function to evaluate the model on the test set
import numpy as np
from sklearn.metrics import (precision_score, recall_score, accuracy_score,
                             f1_score, roc_auc_score,
                             average_precision_score)
from utils.image import ImageUtil


def created_test_set_performance(model, x_test, y_test, distance_cutoff=0.5):
    """
    Evaluate the performance on the manuly created test set
    """

    distances = model.predict([x_test[:, 0], x_test[:, 1]], verbose=0)
    # print(distances)
    pred = distances < distance_cutoff
    precision = precision_score(y_test, pred, average='binary')
    recall = recall_score(y_test, pred, average='binary')
    f1 = f1_score(y_test, pred, average='binary')
    roc = roc_auc_score(y_test, pred)
    pr = average_precision_score(y_test, pred)

    metrics = {'distance_cutoff': distance_cutoff,
               'precision': precision,
               'recall': recall,
               'f1': f1,
               'roc': roc,
               'pr': pr}

    print(metrics)

    return metrics


def evaluate_performance(test, preds, characters=['Mario', 'Frieren', "Lae'zel"]):
    """
    Evaluate the performance of the model on the test set
    """
    metrics = {}

    for class_index, character in zip(range(0, test.shape[0]), characters):
        class_labels = test[:, class_index]
        pred_labels = preds[:, class_index]

        precision = precision_score(
            class_labels, pred_labels, average='binary')
        recall = recall_score(class_labels, pred_labels, average='binary')
        f1 = f1_score(class_labels, pred_labels, average='binary')
        if len(np.unique(class_labels)) > 1:
            roc = roc_auc_score(class_labels, pred_labels)
        else:
            roc = None
        pr = average_precision_score(class_labels, pred_labels)

        metrics[character] = {'precision': round(precision, 2),
                              'recall': round(recall, 2),
                              'f1': round(f1, 2),
                              'roc': round(roc, 2) if roc is not None else None,
                              'pr': round(pr, 2)
                              }

    return metrics


def compute_distances(model, sample_embedding, example_embeddings):
    distances = []
    for example_embedding in example_embeddings:
        # Use the Siamese network to compute the distance
        distance = model.predict(
            [np.array([sample_embedding]), np.array([example_embedding])], verbose=0)[0]

        distances.append(distance[0])
    return distances


def display_character_predictions(model: object, embeddings: list, test_index: list, distance_cutoff=0.8):
    """
    Display the character predictions for the test set
    """
    examples_index = [56340, 30180, 49582]
    characters = ["Mario", "Frieren", "Lae'zel"]
    example_embeddings = [embeddings[idx] for idx in examples_index]
    pred_labels = []
    pred_distances = []
    im = ImageUtil('data/images.tar')

    for idx in test_index:
        temp_label = [0 for _ in range(len(examples_index))]
        distances = compute_distances(
            model, embeddings[idx], example_embeddings)
        pred_distances.append(distances)
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        if min_distance <= distance_cutoff:
            temp_label[min_index] = 1
            print(
                f'pred character: {characters[min_index]}, index: {idx}, distances: {distances}')
            im.display_image(idx)
        pred_labels.append(temp_label)

    return pred_labels, pred_distances
