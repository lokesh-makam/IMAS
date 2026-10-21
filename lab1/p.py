"""
Lab 1 - Knowledge Representation — Semantic Network with Inheritance
U23AI093
"""

class SemanticNetwork:
    def __init__(self):
        self.triples = []

    def add_triple(self, subject, relation, obj):
        self.triples.append((subject, relation, obj))

    def get_inheritance_chain(self, node):
        """
        Repeatedly follow is-a links until no parent exists.
        Returns the ordered list of concepts from most specific to most general.
        """
        chain = [node]
        current = node
        while True:
            # Find parent via 'is-a' relation
            parents = [obj for (subj, rel, obj) in self.triples
                      if subj == current and rel == 'is-a']
            if not parents:
                break

            # Single inheritance: follow the first parent found
            current = parents[0]
            if current in chain: # Prevent cycle infinite loops
                break
            chain.append(current)

        return chain

    def query(self, node, relation):
        chain = self.get_inheritance_chain(node)

        for current_node in chain:
            if relation == 'can':
                exceptions = [obj for (subj, rel, obj) in self.triples
                              if subj == current_node and rel == 'cannot']
                if exceptions:
                    return f"cannot {exceptions[0]}"

            values = [obj for (subj, rel, obj) in self.triples
                      if subj == current_node and rel == relation]
            if values:
                return values[0]

        return "unknown"

    def display_kb(self):
        print("--- Knowledge base Triples ---")
        for subj, rel, obj in self.triples:
            print(f"({subj}, {rel}, {obj})")
        print("------------------------------\n")

class FrameSystem:
    def __init__(self):
        self.frames = {}

    def add_frame(self, concept, slots):
        """Adds a concept frame (dict of slots & fillers)."""
        self.frames[concept] = slots

    def get_inheritance_chain(self, concept):
        chain = [concept]
        current = concept
        while current in self.frames and 'ako' in self.frames[current]:
            parent = self.frames[current]['ako']
            if parent in chain:
                break
            chain.append(parent)
            current = parent
        return chain

    def query(self, concept, slot):
        chain = self.get_inheritance_chain(concept)
        for current in chain:
            if current in self.frames:
                if slot == 'can' and 'cannot' in self.frames[current]:
                    return f"cannot {self.frames[current]['cannot']}"
                if slot in self.frames[current]:
                    return self.frames[current][slot]
        return "unknown"

class MultipleInheritanceNetwork(SemanticNetwork):
    def get_inheritance_chain_bfs(self, node):
        visited = []
        queue = [node]

        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.append(current)
                parents = [obj for (subj, rel, obj) in self.triples
                           if subj == current and rel == 'is-a']
                for parent in parents:
                    if parent not in visited and parent not in queue:
                        queue.append(parent)
        return visited

    def query_multiple(self, node, relation):
        chain = self.get_inheritance_chain_bfs(node)
        for current_node in chain:
            if relation == 'can':
                exceptions = [obj for (subj, rel, obj) in self.triples
                              if subj == current_node and rel == 'cannot']
                if exceptions:
                    return f"cannot {exceptions[0]}"

            values = [obj for (subj, rel, obj)in self.triples 
                      if subj == current_node and rel == relation]
            if values:
                return values[0]
        return "unknown"

def run_experiment():
    print("==========================================================")
    print("EXPERIMENT RESULT: SEMANTIC NETWORK WITH INHERITANCE")
    print("==========================================================\n")

    sn = SemanticNetwork()

    sn.add_triple("animal", "has", "skin")
    sn.add_triple("fish", "is-a", "animal")
    sn.add_triple("bird", "is-a", "animal")
    sn.add_triple("bird", "has", "feathers")
    sn.add_triple("bird", "can", "fly")

    sn.add_triple("shark", "is-a", "fish")
    sn.add_triple("penguin", "is-a", "bird")
    sn.add_triple("penguin", "cannot", "fly")      # Exception to bird
    sn.add_triple("penguin", "can", "swim")        # Local property

    sn.add_triple("ostrich", "is-a", "bird")
    sn.add_triple("ostrich", "cannot", "fly")      # Exception to bird
    sn.add_triple("ostrich", "can", "run fast")    # Local property

    # Display Knowledge Base
    sn.display_kb()

    # Display Inheritance Chains
    print("--- Inheritance Chains Obtained ---")
    print(f"Shark Chain   : {sn.get_inheritance_chain('shark')}")
    print(f"Penguin Chain : {sn.get_inheritance_chain('penguin')}")
    print(f"Ostrich Chain : {sn.get_inheritance_chain('ostrich')}")
    print("-----------------------------------\n")

    # Perform Queries
    print("--- Query Execution Results ---")
    queries = [
        ("shark", "has"),
        ("shark", "can"),
        ("bird", "can"),
        ("penguin", "has"),
        ("penguin", "can"),
        ("ostrich", "can"),
        ("ostrich", "has"),
        ("rock", "has")
    ]

    for subj, rel in queries:
        ans = sn.query(subj, rel)
        print(f"query('{subj}', '{rel}') -> {ans}")
    print("---------------------------------\n")

    # Execute Exercise 2: Frame-Based System Demonstration
    print("==========================================================")
    print("EXERCISE 2: FRAME-BASED SYSTEM EXECUTION")
    print("==========================================================\n")
    fs = FrameSystem()
    fs.add_frame("animal", {"has": "skin"})
    fs.add_frame("bird", {"ako": "animal", "can": "fly", "has": "feathers"})
    fs.add_frame("penguin", {"ako": "bird", "cannot": "fly", "can": "swim"})
    fs.add_frame("ostrich", {"ako": "bird", "cannot": "fly", "can": "run fast"})

    print(f"Frame Query: query('penguin', 'can') -> {fs.query('penguin', 'can')}")
    print(f"Frame Query: query('ostrich', 'can') -> {fs.query('ostrich', 'can')}")
    print(f"Frame Query: query('penguin', 'has') -> {fs.query('penguin', 'has')}\n")

    # Execute Exercise 3: Multiple Inheritance Demonstration
    print("==========================================================")
    print("EXERCISE 3: MULTIPLE INHERITANCE (BFS SEARCH ORDER)")
    print("==========================================================\n")
    minet = MultipleInheritanceNetwork()
    minet.add_triple("pet", "has", "owner")
    minet.add_triple("wild_animal", "can", "hunt")
    minet.add_triple("dog", "is-a", "pet")
    minet.add_triple("wolf", "is-a", "wild_animal")
    # Dog-Wolf hybrid has two parents: dog (pet) and wolf (wild_animal)
    minet.add_triple("hybrid_hound", "is-a", "dog")
    minet.add_triple("hybrid_hound", "is-a", "wolf")

    print(f"Hybrid Hound BFS Chain: {minet.get_inheritance_chain_bfs('hybrid_hound')}")
    print(f"query_multiple('hybrid_hound', 'has') -> {minet.query_multiple('hybrid_hound', 'has')}")
    print(f"query_multiple('hybrid_hound', 'can') -> {minet.query_multiple('hybrid_hound', 'can')}")

if __name__ == "__main__":
    run_experiment()