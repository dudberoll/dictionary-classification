
def clique(graph: dict):

    R = set()
    P = set(graph.keys())
    X = set()
    clique_f = []
    BronKerbosch(R,P,X,graph,clique_f)

    return clique_f

def BronKerbosch(R: set, P: set, X:set, graph:dict, clique:list) -> list:
    """
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
    
my_graph = {
    1: {2, 3},
    2: {1, 4},
    3: {1, 4, 5},
    4: {2, 3, 5},
    5: {3, 4}
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
print(clique(my_graph_2))