
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
        if clique:
            return

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
    
# treshhold 0.8
my_graph = {'break': {'go', 'left', 'left_hand', 'start'},
 'go': {'break', 'left', 'left_hand', 'stop'},
 'left': {'break', 'go', 'start', 'stop'},
 'left_hand': {'break', 'go', 'start', 'stop'},
 'start': {'break', 'left', 'left_hand', 'stop'},
 'stop': {'go', 'left', 'left_hand', 'start'}
 }

my_graph_2 = {
    1: {2, 3, 4},
    2: {1, 3, 4},
    3: {1, 2, 4, 5},
    4: {1, 2, 3, 5},
    5: {3, 4, 6},
    6: {5}
}
print("dsf", clique(my_graph))



def format_algo_graph_(edge_weights):
    """
    Создает граф на основе весов ребер.

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