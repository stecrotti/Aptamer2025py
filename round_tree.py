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
        assert(v < self.get_n_nodes()), f"v={v}"
        return self.ancestors_flat[self.offset[v] : self.offset[v] + self.length[v]]

    def get_n_nodes(self):   # excluding the root
        return len(self.parent)

    def get_depth(self):
        return self.length.max().item()

    def parent(self, v):
        assert(v < self.get_n_nodes())
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
        

class RoundTree:
    def __init__(
        self,
        n_modes: int | None = None,
        tree: Tree | None = None,  
        selected_modes: torch.BoolTensor | None = None,   # (n_rounds * n_modes) modes selected for at each round
    ):
        if n_modes is None:
            if selected_modes is None:
                raise ValueError("Must provide either the selected modes or the total number of modes")
            else:
                n_modes = selected_modes.size(1)
        if tree is None:
            tree = Tree()
        if selected_modes is None:
            selected_modes = torch.BoolTensor()
        elif not (selected_modes.sum(dim=1) > 0).all():
            raise ValueError("At least one mode must be selected")
            
        n_rounds = tree.parent.size(0)
        assert(selected_modes.size(0) == n_rounds)

        self.n_modes = n_modes
        self.tree = tree
        self.selected_modes = selected_modes   

    def add_node(self, parent_node, selected_modes, name = None):
        assert selected_modes.size(0) == self.n_modes, f"Number of modes in `selected_modes`, {selected_modes.size(0)}, different from the expected {self.n_modes}"
        assert selected_modes.sum() > 0, f"At least one mode must be selected"
        self.tree.add_node(parent_node, name=name)
        self.selected_modes = torch.cat((self.selected_modes, selected_modes[None,:]))

    def parent(self, t):
        return self.tree.parent(t)

    def ancestors_of(self, v):
        return self.tree.ancestors_of(v)

    def get_n_modes(self):
        return self.n_modes