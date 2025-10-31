
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from typing import Dict, List, Any, Tuple

class SignalDataPreprocessor(TransformerMixin):
    """
    здесь потом будет обработка сигнала. 
    сигнал ЭЭГ -> признаки
    """
    def __init__(self, scaler):
        
        self.scaler = StandardScaler()


    def fit(self, X, y):

        self.scaler.fit(X)

        return self

    def transform(self, X) -> np.ndarray:

        X_scaled = self.scaler.transform(X)
        return X_scaled

def generate_synthetic_data():
    """

    Сгенерированные данные с помощью ГПТ только для ознакомительного примера.

    Returns:
        - df: concat( X_features, y, equivalence_classes) .
        - X: признаки сигналов с ЭЭГ.
        - y: слова (действия).
        - equivalence_classes: словарь классов эквивалентности.
    """
    X: np.ndarray = np.array([
        [15.2, 8.1, 3.5, 22.9], [14.9, 9.0, 4.1, 21.5], [15.054, 8.769, 3.833, 21.405], [14.525, 9.321, 3.918, 21.953], [14.780, 7.907, 3.965, 23.141], # старт
        [14.8, 8.5, 3.7, 22.0], [15.1, 8.0, 3.6, 22.8], [14.6, 9.1, 4.0, 21.7], # старт (новые)
        [15.1, 8.3, 3.8, 22.5], [14.7, 8.9, 4.0, 21.9], [15.5, 8.0, 3.4, 23.1], [15.312, 8.156, 3.611, 22.889], [14.995, 8.651, 4.198, 21.655], # вперед
        [15.0, 8.4, 3.9, 22.2], [15.2, 8.1, 3.5, 22.7], [14.8, 8.8, 4.1, 21.8], # вперед (новые)
        [5.3, 12.1, 18.2, 7.7], [4.8, 11.5, 19.0, 8.1], [4.348, 11.969, 19.383, 8.071], [4.489, 11.137, 19.496, 8.272], [4.838, 12.025, 17.903, 7.905], # стоп
        [5.0, 11.8, 18.5, 8.0], [4.5, 11.3, 19.2, 8.2], [5.2, 12.0, 18.1, 7.8], # стоп (новые)
        [5.1, 12.5, 18.0, 7.9], [4.9, 11.9, 18.8, 8.3], [5.5, 12.0, 19.1, 8.0], [5.099, 12.451, 18.150, 7.850], [5.411, 12.210, 19.055, 8.150], # пауза
        [5.2, 12.3, 18.3, 8.1], [4.8, 11.8, 18.9, 8.2], [5.3, 12.1, 19.0, 7.9], # пауза (новые)
        [9.2, 14.8, 9.5, 15.5], [8.8, 15.2, 10.5, 14.5], [8.950, 15.050, 9.950, 15.000], [9.100, 14.700, 9.700, 15.300], [9.000, 15.100, 10.300, 14.800], # левая_рука
        [9.0, 14.9, 10.0, 15.1], [8.9, 15.0, 10.1, 14.9], [9.1, 14.8, 9.8, 15.2], # левая_рука (новые)
        [9.1, 15.0, 9.8, 15.2], [8.9, 14.9, 10.2, 14.9], [9.000, 14.950, 10.000, 15.050], [9.250, 15.150, 9.550, 15.350], [8.850, 14.800, 10.350, 14.750], # влево
        [9.0, 15.1, 9.9, 15.0], [8.8, 14.9, 10.1, 14.8], [9.2, 15.0, 9.7, 15.3]  # влево (новые)
    ])

    y: np.ndarray = np.array([
        "start", "start", "start", "start", "start",
        "start", "start", "start", # новые
        "go", "go", "go", "go", "go",
        "go", "go", "go", # новые
        "stop", "stop", "stop", "stop", "stop",
        "stop", "stop", "stop", # новые
        "break", "break", "break", "break", "break",
        "break", "break", "break", # новые
        "left_hand", "left_hand", "left_hand", "left_hand", "left_hand",
        "left_hand", "left_hand", "left_hand", # новые
        "left", "left", "left", "left", "left",
        "left", "left", "left" # новые
    ])

    equivalence_classes = {
        'start': 1, 'go': 1, 'stop': 2, 'break': 2, 'left_hand': 3, 'left': 3
    }

    # добавляем шум
    _noise: float = 5.0
    noise: np.ndarray = np.random.normal(0, _noise, size=X.shape)
    X += noise

    # определяем в каком классе экв. лежит слово
    num_eq_classes = [equivalence_classes[label] for label in y]

    df: pd.DataFrame = pd.DataFrame({
        "X": [x for x in X],
        "y": y,
        "num_eq_classes": num_eq_classes
    })

    return df, X, y, equivalence_classes

def preprocess_data(X: np.ndarray) -> np.ndarray:
    
    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)
    return X_scaled

def train_binary_svm(X_train: np.ndarray, y_train: np.ndarray, class1: str, class2: str):
    """
    Обучает бинарную модель SVM для двух указанных классов

    Args:
        X_train: признаки
        y_train: слова
        class1: метка первого класса из пары
        class2: метка второго класса из пары

    Returns:
        cls: веса svc
    """
    train_mask: np.ndarray = (y_train == class1) | (y_train == class2)
    X_train_filtered: np.ndarray = X_train[train_mask]
    y_train_filtered: np.ndarray = y_train[train_mask]

    cls = make_pipeline(StandardScaler(),
                        SVC(kernel="linear", C=1, random_state=42))
    cls.fit(X_train_filtered, y_train_filtered)
    return cls

def create_graph(classification_scores: dict) -> nx.Graph:
    """
    Создаёт граф networkx из значений точности классификации между парами классов.

    Args:
        classification_scores: Словарь ключи — кортежи пар слов,
                               значения —  метрики классификации.

    Returns:
        G: Созданный граф networkx.
    """
    G = nx.Graph()
    for e, score in classification_scores.items():
        G.add_edge(e[0], e[1], weight=score)
    return G

def visualize_graph(graph: nx.Graph) -> None:
    """
    Визуализирует граф networkx пар слов и их метрик классификации.

    Args:
        graph: граф networkx
    """
    plt.figure(figsize=(10, 8)) 
    pos = nx.spring_layout(graph) 
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()

if __name__ == "__main__":

    df, X, y, equivalence_classes = generate_synthetic_data()
    X_scaled = preprocess_data(X)

    unique_labels = df['y'].unique()

    # 5. все уникальные пары, где слова принадлежат разным классам эквивалентности
    all_pairs = list(itertools.combinations(unique_labels, 2))
    filtered_pairs = []
    for pair in all_pairs:
        num_class_in_pair_1 = df[df['y'] == pair[0]]['num_eq_classes'].iloc[0]
        num_class_in_pair_2 = df[df['y'] == pair[1]]['num_eq_classes'].iloc[0]
        print(num_class_in_pair_1, num_class_in_pair_2)

        if num_class_in_pair_1 == num_class_in_pair_2:
            all_pairs.remove(pair)


    classification_scores = {}

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # считаем svc для каждого класса
    for pair in all_pairs:
        class1, class2 = pair
        cls = train_binary_svm(X_train, y_train, class1, class2)

        test_mask = (y_test == class1) | (y_test == class2)
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        y_pred = cls.predict(X_test_filtered)

        classification_scores[pair] = accuracy_score(y_test_filtered, y_pred)

    
    print("Точность accuracy для каждой пары")
    for pair, score in classification_scores.items():
        print(f"{pair}: accuracy: {score:.2f}")

    classification_graph = create_graph(classification_scores)
    visualize_graph(classification_graph)

