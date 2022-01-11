class DisjointSet:
    def __init__(self, elements, parent):
        self.elements = elements
        self.parent = parent

    def find(self, item):
        if self.parent[item] == item:
            return item
        else:
            res = self.find(self.parent[item])
            self.parent[item] = res
            return res

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        self.parent[root1] = root2


def disjoint_set_union(groups: "list[tuple]"):
    elements = list(set(sum(map(list, groups), list())))
    parents = dict(zip(elements, elements))
    ds = DisjointSet(elements, parents)
    
    for grp in groups:
        for elem in grp[1:]:
            ds.union(grp[0], elem)

    root_to_group = dict()
    for elem in elements:
        root = ds.find(elem)
        if root not in root_to_group:
            root_to_group[root] = list()
        root_to_group[root].append(elem)

    groups = list(root_to_group.values())
    return groups


def toposort(vertices: list, edges):
    def _toposort(v, visited, sorted_vs):
        visited[v] = True
        if v in edges:
            for u in edges[v]:
                if visited[u]: continue
                _toposort(u, visited, sorted_vs)
        sorted_vs.append(v)

    visited = dict.fromkeys(vertices, False)
    sorted_vs = list()

    for v in vertices:
        if visited[v]: continue
        _toposort(v, visited, sorted_vs)

    return sorted_vs[::-1]


def same_padding(length, kernel_size, stride):
    if length % stride == 0:
        tot_pad = max(kernel_size - stride, 0)
    else:
        tot_pad = max(kernel_size - (length % stride), 0)

    pad_left = tot_pad // 2
    pad_right = tot_pad - pad_left

    return pad_left, pad_right


if __name__ == "__main__":
    vertices = ['a', 'b', 'c', 'd']
    edges = {
        'a': ['b', 'c'],
        'b': ['d'],
        'c': ['d'],
        'd': [],
    }
    print(toposort(vertices, edges))
