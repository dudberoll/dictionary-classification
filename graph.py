import networkx as nx
import matplotlib.pyplot as plt

def clique(graph: dict):

    R = set()
    P = set(graph.keys())
    X = set()
    clique_f = []
    BronKerbosch(R,P,X,graph,clique_f)

    return clique_f

def BronKerbosch(R: set, P: set, X:set, graph:dict, clique:list) -> list:
    """
    Алгоритм для поиска полного подграфа (в нашем случаем первого)
    R := это множество вершин максимальной клики.
    P := множество возможных вершин максимальной клики.
    X := множество исключенных вершин.
    """
    
    N = dict()
    if not P and not X:
        clique.append(R)
        return 
    
    """
    R ⋃ {v} := соединение R с единичным множеством (singleton) v.
    P ⋂ N(v) := пересечение множества P с соседями v.
    X ⋂ N(v) := пересечение множества X с соседями v.
    P \ {v} := относительное соотнесение P единичного множества v.
    X ⋃ {v} := соединение множества X и единичного набора v.
    """
    for v in list(P):
        # if clique:
        #     return

        N_v = graph.get(v, set())    
        BronKerbosch(
            R | {v},
            P=P.intersection(N_v),
            X=X.intersection(N_v),
            graph=graph,
            clique=clique
        )
        P.remove(v)
        X.add(v)


def format_algo_graph_(edge_weights):
    """
    Форматирует граф для алгоритма 

    Args:
        edge_weights: Словарь, ключи - веса
        my_graph: граф в формате словаря 
    """

    all_nodes = set()
    for edge in edge_weights.keys():
        all_nodes.add(edge[0])
        all_nodes.add(edge[1])
    all_nodes = sorted(list(all_nodes))

    my_graph = {}
    for node_label in all_nodes:
        my_graph[node_label] = set() 

    # заполняем граф вершинами
    my_graph = {}
    for node_label in all_nodes:
        my_graph[node_label] = set()

    # заполняем граф ребрами 
    for (node1_label, node2_label), weight in edge_weights.items():
        my_graph[node1_label].add(node2_label)
        my_graph[node2_label].add(node1_label) # в две стороны ребра

    return my_graph

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

def filter_dict_by_weight_threshold(input_dict, threshold):

    filtered = {}
    for edge, weight in input_dict.items():
        if weight >= threshold:
            filtered[edge] = weight
    return filtered

def find_clique(edge_weights, target_k, start_threshold=0.8, step=0.2, min_threshold=0.0):
    """
    Итеративно понижает порог и строит граф до тех пор, пока размер максимальной клики
    не станет равен target_k. Возвращает"""
    threshold = start_threshold
    while threshold >= min_threshold:
        filtered_edges = filter_dict_by_weight_threshold(edge_weights, threshold)
        algo_graph = format_algo_graph_(filtered_edges)
        cliques = clique(algo_graph)
        max_cl = max(cliques, key=len) if cliques else set()
        if len(max_cl) == target_k:
            return max_cl
        threshold -= step
    return None

 # демонстрационный код перенесён в main.py