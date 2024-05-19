import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(42)


def create_label_set(test_index: list, embeddings: dict, n: int = 1000, threshold_character: int = 0.8, threshold_others: int = 0.4) -> dict:
    """
    Classify images using similarity-based approach (cosine similarity)

    """
    characters = ["Mario", "Frieren", "Lae'zel"]
    examples_index = [56340, 30180, 49582]

    label_set = {'Mario': [], 'Frieren': [],
                 "Lae'zel": [], "Others": [], "Uncertain": []}

    for index in range(0, n):
        if index not in test_index:
            similarities = {}
            for character, ex_index in zip(characters, examples_index):
                similarity = cosine_similarity(embeddings[index].reshape(
                    1, -1), embeddings[ex_index].reshape(1, -1))[0][0]
                similarities[character] = similarity
            max_similarity = max(similarities.values())
            if max_similarity >= threshold_character:
                predicted_label = max(similarities, key=similarities.get)
                if predicted_label == "Mario":
                    label_set['Mario'].append((max_similarity, index))
                elif predicted_label == "Frieren":
                    label_set['Frieren'].append((max_similarity, index))
                elif predicted_label == "Lae'zel":
                    label_set["Lae'zel"].append((max_similarity, index))
            elif max_similarity <= threshold_others:
                label_set["Others"].append((max_similarity, index))
            else:
                label_set["Uncertain"].append((max_similarity, index))

    return label_set


def select_train_set(label_set: dict) -> list:
    """
    Select the most likley similar character in the character sets and most unlikely for the others
    """
    characters_max_cap = 200
    others_max_cap = 500

    m_set = [index for _, index in sorted(label_set['Mario'], reverse=True)[
        :characters_max_cap]]
    f_set = [index for _, index in sorted(label_set['Frieren'], reverse=True)[
        :characters_max_cap]]
    l_set = [index for _, index in sorted(label_set["Lae'zel"], reverse=True)[
        :characters_max_cap]]
    # choose small distance as the most unlikely
    o_set = [index for _, index in sorted(
        label_set["Others"], reverse=False)[:others_max_cap]]

    print(len(m_set), len(f_set), len(l_set), len(o_set))

    train_set = []

    for idx in m_set:
        train_set.append((idx, 'm'))
    for idx in f_set:
        train_set.append((idx, 'f'))
    for idx in l_set:
        train_set.append((idx, 'l'))
    for idx in o_set:
        train_set.append((idx, 'o'))

    train_set

    return train_set


def make_pairs(train_labels: list, embeddings: dict, characters: list = ['m', 'f', 'l']):
    """
    Create pairs of embeddings and labels for training
    """
    pairs_index = []
    pairs_embedding = []
    labels = []

    for i, label_1 in train_labels:
        for j, label_2 in train_labels:
            if label_1 in characters and label_1 == label_2:
                labels.append(1)
            else:
                labels.append(0)

            pairs_index.append([i, j])
            pairs_embedding.append([embeddings[i], embeddings[j]])

    return pairs_index, np.array(pairs_embedding), np.array(labels)


def create_manuel_labels():
    N = 10000
    m_set = [56340, 4, 5, 6, 10, 23, 82, 59, 95, 148, 127, 7270]
    f_set = [30180, 238, 846, 861, 1196, 1237, 2191, 2326, 2405, 3229, 5191]
    l_set = [13459, 40591, 49582]

    # o_set = [idx for idx in np.random.randint(
    #     0, N, 30) if idx not in m_set and idx not in f_set and idx not in l_set]

    # using random selection above after manual inspection,
    # excluding the three characters, the following indices are selected
    o_set = [860, 5390,  5734, 466, 4426, 5578, 8322, 769, 6949, 2433, 5311,
             5051, 6420, 1184, 4555, 3385, 6396, 8666, 9274, 2558, 7849, 2047, 2747, 9167, 9998, 189]

    print(len(o_set))

    train_set = []

    for idx in m_set:
        train_set.append((idx, 'm'))
    for idx in f_set:
        train_set.append((idx, 'f'))
    for idx in l_set:
        train_set.append((idx, 'l'))
    for idx in o_set:
        train_set.append((idx, 'o'))

    return train_set
