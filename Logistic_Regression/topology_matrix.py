    def topology_matrix(self, topology: str, n_cores: int, p: float = 0.3) -> np.ndarray:

        # === Ring topology ===
        if topology == 'ring':
            W = np.zeros((n_cores, n_cores))
            value = 1.0 / 3 if n_cores >= 3 else 1.0 / 2
            for i in range(n_cores):
                W[i, i] = value
                W[i, (i - 1) % n_cores] = value
                W[i, (i + 1) % n_cores] = value
            G = nx.cycle_graph(n_cores)
            comment = f"[Ring] Each node communicates with its two neighbors and itself."

        # === Centralized topology ===
        elif topology == 'centralized':
            W = np.ones((n_cores, n_cores), dtype=np.float64) / n_cores
            G = nx.complete_graph(n_cores)
            comment = f"[Centralized] Fully connected network with uniform averaging."

        # === Disconnected topology ===
        elif topology == 'disconnected':
            W = np.eye(n_cores, dtype=np.float64)
            G = nx.empty_graph(n_cores)
            comment = f"[Disconnected] No communication between nodes, only self-loops."

        # === Torus topology ===
        elif topology == 'torus':
            assert int(np.sqrt(n_cores)) ** 2 == n_cores, "n_cores must be a perfect square for torus"
            side = int(np.sqrt(n_cores))
            G = nx.generators.lattice.grid_2d_graph(side, side, periodic=True)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            W = nx.adjacency_matrix(G).toarray().astype(float)
            np.fill_diagonal(W, 1.0)
            W = W / 5.0  # each node connects to 4 neighbors + self-loop
            comment = f"[Torus] 2D grid {side}x{side} with periodic boundaries. Each node connects to 4 neighbors + itself."

        # === Erdős–Rényi (ER) random graph ===
        elif topology == 'er':
            G = nx.erdos_renyi_graph(n_cores, p)
            W = nx.adjacency_matrix(G).toarray().astype(float)
            np.fill_diagonal(W, 1.0)  # add self-loops
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # avoid division by zero
            W = W / row_sums
            comment = f"[ER] Erdős–Rényi random graph G({n_cores}, {p}). Each edge appears with probability {p}."

        else:
            raise ValueError(f"Unsupported topology: {topology}")

        # === Print comment ===
        print(comment)

        # === Visualize the topology ===
        plt.figure(figsize=(4, 4))
        nx.draw(
            G, with_labels=True, node_color="skyblue",
            node_size=800, font_size=10, font_weight="bold"
        )
        plt.title(f"Topology: {topology}")
        plt.show()

        return W
