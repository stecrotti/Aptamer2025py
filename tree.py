import torch

def store_ancestors(parent):
    N = parent.size(0)

    ancestors = []
    offset = torch.zeros(N, dtype=torch.long)
    length = torch.zeros(N, dtype=torch.long)

    cur_offset = 0

    for v in range(N):
        path = [v]
        p = parent[v]

        while p != -1:
            path.append(p)
            p = parent[p]

        offset[v] = cur_offset
        length[v] = len(path)
        ancestors.extend(path)
        cur_offset += len(path)

    ancestors_flat = torch.tensor(ancestors, dtype=torch.long)
    return ancestors_flat, offset, length

def ancestors_of(v, ancestors_flat, offset, length):
    return ancestors_flat[offset[v] : offset[v] + length[v]]


class Tree:
    def __init__(self,
                 parent: torch.Tensor | None = None,  # parent[v] is the index of v's parent. the root is -1
                 nodename = None, 
                ):
        if parent is None:
            parent = torch.IntTensor()
            
        assert((parent >= -1).all())
        if nodename is None:
            nodename = [str(v) for v in range(len(parent))]
        assert(len(nodename) == parent.size(0))
        self.parent = parent
        self.nodename = nodename
        self.ancestors_flat, self.offset, self.length = store_ancestors(parent)

    def ancestors_of(self, v):  # returns a torch vector with v and its ancestors
        assert(v < self.get_n_nodes()), f"Queried ancestors of node v={v}, but node indices are 0:{self.get_n_nodes()-1}"
        return self.ancestors_flat[self.offset[v] : self.offset[v] + self.length[v]]

    def get_n_nodes(self):   # excluding the root
        return len(self.parent)

    def get_depth(self):
        return self.length.max().item()

    def get_parent(self, v):
        assert(v < self.get_n_nodes() and v >= 0), f"Node index {v} not in range"
        return self.parent[v]

    def add_node(self, parent_node, name = None):
        N = self.get_n_nodes
        if name is None:
            name = str(N)
        if type(parent_node) is str:
            if parent_node == "root":
                parent_node = -1
            else:
                parent_node = self.nodename.index(parent_node)
            
        self.parent = torch.cat((self.parent, torch.IntTensor([parent_node])))
        self.nodename.append(name)
        self.ancestors_flat, self.offset, self.length = store_ancestors(self.parent)  # re-computing everything, not efficient but ok  

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False