from graph import *


# TODO: move compute definition to utils.py
def test_graph1():
    graph = Subgraph()
    X = graph.new_input([32, 28, 28], 'X')
    Y, W = graph.linear_map(X, [64, 14, 14], 'Y', 'W')
    
    print('Compute Definition:')
    print(graph.state.stages[0].compute_expr)
    
    return graph.state


def test_graph2():
    graph = Subgraph()
    X1 = graph.new_input([32, 28, 28], 'X1')
    X2 = graph.new_input([32, 28, 28], 'X2')
    Y1, W1 = graph.linear_map(X1, [64, 14, 14], 'Y1', 'W1')
    Y2, W2 = graph.linear_map(X2, [64, 14, 14], 'Y2', 'W2')
    O, W3 = graph.bilinear_map(Y1, Y2, [64, 7, 7], 'O', 'W3')

    state = graph.state
    
    print('Compute Definition:')
    for stage in state.stages: 
        print(stage.compute_expr)

    return state


if __name__ == '__main__':
    test_graph1()
    test_graph2()
