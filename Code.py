'''
Team Members:
    CS24MTECH14013 : Mohit Manoj Patil
    CS24MTECH14010: Veeresh Shukla
    CS24MTECH12018 : Deeba Afridi
'''

import pandas as pd
import networkx as nx
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import warnings
import os # Import os for path joining

# Suppress KMeans FutureWarning about n_init
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

print("--- Node2Vec Clustering Assignment (Elbow Method - Alternative Implementation - Direct Values) ---")

# --- Simpler Classes ---

class DataProcessor:
    """Handles data loading and basic processing."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_process(self):
        print(f"\n[1/6] Loading and processing data from '{self.file_path}'...")
        start_time = time.time()
        try:
            data_frame = pd.read_excel(self.file_path)
            print(f"Successfully loaded {len(data_frame)} transactions.")
            required_cols = ['Sender', 'Receiver', 'Amount']
            if not all(col in data_frame.columns for col in required_cols):
                raise ValueError(f"Excel file must contain columns: {required_cols}")
            data_frame['Sender'] = data_frame['Sender'].astype(str)
            data_frame['Receiver'] = data_frame['Receiver'].astype(str)
            data_frame['Amount'] = pd.to_numeric(data_frame['Amount'], errors='coerce')
            data_frame.dropna(subset=['Sender', 'Receiver', 'Amount'], inplace=True)
            data_frame['Amount'] = data_frame['Amount'].astype(float)

        except FileNotFoundError:
            print(f"Error: '{self.file_path}' not found.")
            print("Creating dummy data for demonstration purposes...")
            num_dummy_nodes = 150
            num_dummy_transactions = 800
            senders = [str(random.randint(1, num_dummy_nodes)) for _ in range(num_dummy_transactions)]
            receivers = [str(random.randint(1, num_dummy_nodes)) for _ in range(num_dummy_transactions)]
            receivers = [r if r != s else str((int(r) % num_dummy_nodes) + 1) for s, r in zip(senders, receivers)]
            amounts = np.random.randint(10, 1000, num_dummy_transactions).astype(float)
            data_frame = pd.DataFrame({'Sender': senders, 'Receiver': receivers, 'Amount': amounts})
            print(f"Generated {len(data_frame)} dummy transactions.")
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            raise

        print(f"Data loading finished in {time.time() - start_time:.2f} seconds.")
        return data_frame

class GraphCreator:
    """Creates and aggregates the graph."""
    def build_and_aggregate_graph(self, data_frame, k_range_for_check):
        print("\n[2/6] Creating and aggregating the graph...")
        start_time = time.time()

        transaction_graph = nx.DiGraph()
        edge_weight_aggregation = defaultdict(float)

        for _, row in data_frame.iterrows():
            sender_id = row['Sender']
            receiver_id = row['Receiver']
            transaction_amount = row.get('Amount', 1.0)
            if sender_id != receiver_id:
                edge_weight_aggregation[(sender_id, receiver_id)] += transaction_amount

        all_involved_nodes = set(data_frame['Sender']) | set(data_frame['Receiver'])
        transaction_graph.add_nodes_from(all_involved_nodes)

        for (u, v), total_weight in edge_weight_aggregation.items():
            if total_weight > 0:
                transaction_graph.add_edge(u, v, weight=total_weight)

        isolated_nodes = list(nx.isolates(transaction_graph))
        transaction_graph.remove_nodes_from(isolated_nodes)
        if isolated_nodes:
             print(f"Removed {len(isolated_nodes)} isolated nodes after aggregation.")

        graph = transaction_graph
        node_list = list(graph.nodes())
        num_graph_nodes = graph.number_of_nodes()
        num_graph_edges = graph.number_of_edges()
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_list)}
        index_to_node_id = {i: node_id for node_id, i in node_id_to_index.items()}

        print(f"Graph created/aggregated.")
        print(f"Number of nodes: {num_graph_nodes}")
        print(f"Number of edges (aggregated): {num_graph_edges}")

        min_k_start = k_range_for_check.start
        if num_graph_nodes < min_k_start:
             print(f"Error: Number of nodes ({num_graph_nodes}) is less than the minimum K ({min_k_start}) tested for the Elbow method. Cannot proceed.")
             return None, None, None, None, None, False
        if num_graph_nodes < 2:
            print(f"Error: Graph has only {num_graph_nodes} node(s). Cannot perform clustering.")
            return None, None, None, None, None, False

        if num_graph_edges == 0 and num_graph_nodes > 1:
            print("Warning: Graph has nodes but no edges. Node2Vec will not work, but clustering might still be attempted on isolated points if > 1 node.")
        elif num_graph_edges == 0 and num_graph_nodes <= 1:
             print("Error: Graph has no edges or only one node. Cannot perform Node2Vec or clustering.")
             return None, None, None, None, None, False

        print(f"Graph processing finished in {time.time() - start_time:.2f} seconds.")
        return graph, node_list, node_id_to_index, index_to_node_id, num_graph_nodes, True

# --- Refactored AliasSampler and Walk Generation ---

class AliasMethod:
    """Handles alias method setup and drawing samples."""
    def setup_table(self, probabilities):
        """Compares utility lists for non-uniform sampling."""
        prob_len = len(probabilities)
        if prob_len == 0:
            return [], []

        adjusted_probs = [0.0] * prob_len
        alias_table = [0] * prob_len

        small_indices = []
        large_indices = []
        for k, prob_val in enumerate(probabilities):
            adjusted_probs[k] = prob_len * prob_val
            if adjusted_probs[k] < 1.0:
                small_indices.append(k)
            else:
                large_indices.append(k)

        while small_indices and large_indices:
            small_idx = small_indices.pop()
            large_idx = large_indices.pop()
            alias_table[small_idx] = large_idx
            adjusted_probs[large_idx] = adjusted_probs[large_idx] + adjusted_probs[small_idx] - 1.0
            if adjusted_probs[large_idx] < 1.0:
                small_indices.append(large_idx)
            else:
                large_indices.append(large_idx)

        # Numerical stability correction
        adjusted_probs = [min(1.0, p) for p in adjusted_probs]

        return alias_table, adjusted_probs

    def draw_sample(self, alias_table, adjusted_probs):
        """Draws sample using alias sampling."""
        prob_len = len(alias_table)
        if prob_len == 0:
            return -1
        k_index = int(random.random() * prob_len)
        if random.random() < adjusted_probs[k_index]:
            return k_index
        else:
            return alias_table[k_index]

class TransitionCalculator:
    """Calculates transition probabilities and sets up alias tables."""
    def __init__(self, graph, p_param, q_param):
        self.graph = graph
        self.p = p_param
        self.q = q_param
        self.alias_method = AliasMethod()

    def get_distribution(self, current_node, previous_node=None):
        """Calculates alias table for transitions."""
        if previous_node is None:
            neighbors = list(self.graph.neighbors(current_node))
            valid_neighbors = [n for n in neighbors if self.graph[current_node][n].get('weight', 1.0) > 0]
            if not valid_neighbors: return None, []

            unnorm_probs = [self.graph[current_node][n].get('weight', 1.0) for n in valid_neighbors]
            norm_const = sum(unnorm_probs)
            probs = [p / norm_const for p in unnorm_probs] if norm_const > 0 else []
            return self.alias_method.setup_table(probs), valid_neighbors
        else:
            neighbors = list(self.graph.neighbors(current_node))
            valid_neighbors = []
            unnorm_probs = []

            if not neighbors: return None, []

            for neighbor in neighbors:
                weight = self.graph[current_node][neighbor].get('weight', 1.0)
                if weight <= 0: continue

                valid_neighbors.append(neighbor)
                if neighbor == previous_node:
                    unnorm_probs.append(weight / self.p)
                elif self.graph.has_edge(previous_node, neighbor):
                    unnorm_probs.append(weight)
                else:
                    unnorm_probs.append(weight / self.q)

            norm_const = sum(unnorm_probs)
            probs = [p / norm_const for p in unnorm_probs] if norm_const > 0 else []
            return self.alias_method.setup_table(probs), valid_neighbors

def precompute_transition_probs(graph, p_param, q_param):
    """Precomputes alias tables for all possible transitions."""
    print("  Precomputing transition probabilities...")
    node_alias_tables = {}
    edge_alias_tables = {}
    transition_calculator = TransitionCalculator(graph, p_param, q_param)

    for node in graph.nodes():
        alias_setup_node, neighbors_node = transition_calculator.get_distribution(node, None)
        node_alias_tables[node] = (alias_setup_node, neighbors_node)

        if neighbors_node:
            for neighbor in neighbors_node:
                alias_setup_edge, neighbors_neighbor = transition_calculator.get_distribution(neighbor, node)
                edge_alias_tables[(node, neighbor)] = (alias_setup_edge, neighbors_neighbor)
    return node_alias_tables, edge_alias_tables

def simulate_single_walk_step(current_node_id, previous_node_id, node_alias_tables, edge_alias_tables):
    """Simulates a single step in a walk."""
    alias_method = AliasMethod()
    if previous_node_id is None:
        alias_info = node_alias_tables.get(current_node_id)
    else:
        alias_info = edge_alias_tables.get((previous_node_id, current_node_id))

    if alias_info is None or alias_info[0] is None or len(alias_info[1]) == 0:
        return None # No valid next step

    (alias_tab, adjusted_probs), potential_next_node_ids = alias_info
    next_node_idx_in_list = alias_method.draw_sample(alias_tab, adjusted_probs)

    if next_node_idx_in_list != -1:
        return potential_next_node_ids[next_node_idx_in_list]
    else:
        return None # Should not happen if potential_next_node_ids is not empty


def simulate_single_walk(start_node_id, seq_length, node_alias_tables, edge_alias_tables):
    """Simulates a single biased random walk starting from a node ID."""
    current_walk_ids = [start_node_id]

    while len(current_walk_ids) < seq_length:
        current_step_node_id = current_walk_ids[-1]
        previous_step_node_id = current_walk_ids[-2] if len(current_walk_ids) > 1 else None

        next_node_id = simulate_single_walk_step(
            current_step_node_id, previous_step_node_id, node_alias_tables, edge_alias_tables
        )

        if next_node_id is not None:
            current_walk_ids.append(next_node_id)
        else:
             break # No valid next step, stop walk

    return current_walk_ids

def generate_random_walks(graph, num_walks_per_node, seq_length, node_list, node_to_index_map, p_param, q_param):
    """Generates multiple biased random walks."""
    generated_walks_ids = []
    shuffled_nodes = list(node_list)

    node_alias_tables, edge_alias_tables = precompute_transition_probs(graph, p_param, q_param)

    print(f"  Simulating {num_walks_per_node} walks of length {seq_length} for each node...")
    walk_counter = 0
    simulation_start_time = time.time()
    expected_total_walks = len(node_list) * num_walks_per_node

    for walk_iter in range(num_walks_per_node):
        random.shuffle(shuffled_nodes)
        for starting_node_id in shuffled_nodes:
            single_walk_ids = simulate_single_walk(starting_node_id, seq_length, node_alias_tables, edge_alias_tables)

            if len(single_walk_ids) > 1:
                generated_walks_ids.append(single_walk_ids)

            walk_counter += 1
            if walk_counter % 1000 == 0 or walk_counter == expected_total_walks:
                elapsed_sim_time = time.time() - simulation_start_time
                est_total_sim_time = (elapsed_sim_time / walk_counter * expected_total_walks) if walk_counter > 0 else 0
                print(f"    Generated {walk_counter}/{expected_total_walks} walks... (Est. total time: {est_total_sim_time:.1f}s)", end='\r')

    print(f"\n    Walk simulation finished. Generated {len(generated_walks_ids)} valid walks.")
    return generated_walks_ids


def convert_walks_to_indices(walks_ids, node_to_index_map):
    """Converts walks of string IDs to walks of integer indices."""
    indexed_walks = []
    for walk_ids in walks_ids:
        indexed_walk = [node_to_index_map[node_id] for node_id in walk_ids]
        indexed_walks.append(indexed_walk)
    return indexed_walks


class NodeEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(NodeEmbeddingModel, self).__init__()
        self.node_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.kaiming_uniform_(self.node_embeddings.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.context_embeddings.weight, nonlinearity='relu')

    def forward(self, focus_nodes_indices, context_nodes_indices, negative_samples_indices):
        focus_emb = self.node_embeddings(focus_nodes_indices)
        pos_context_emb = self.context_embeddings(context_nodes_indices.squeeze(1))
        neg_context_emb = self.context_embeddings(negative_samples_indices)

        pos_dot_product = torch.sum(focus_emb * pos_context_emb, dim=1)
        pos_loss = F.logsigmoid(pos_dot_product)

        neg_dot_products = torch.bmm(neg_context_emb, focus_emb.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_dot_products).sum(dim=1)

        loss = -(pos_loss + neg_loss)
        return loss.mean()

class BatchGenerator:
    """Generates batches for training."""
    def __init__(self, walks_as_indices, vocab_size, window_size, batch_size, num_negative):
        self.walks_as_indices = walks_as_indices
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_negative = num_negative

    def generate_training_batches(self):
        data_pairs_indices = self.generate_context_pairs()
        if not data_pairs_indices:
            return # Yields nothing

        random.shuffle(data_pairs_indices)
        yield from self.create_batch_tensors(data_pairs_indices)

    def generate_context_pairs(self):
        data_pairs_indices = []
        for walk_sequence_indices in self.walks_as_indices:
            for i, target_node_index in enumerate(walk_sequence_indices):
                pairs = self.extract_context_pairs(walk_sequence_indices, i, target_node_index)
                data_pairs_indices.extend(pairs)
        return data_pairs_indices

    def extract_context_pairs(self, walk_sequence, target_position, target_node_index):
        current_window = random.randint(1, self.window_size)
        start_idx = max(0, target_position - current_window)
        end_idx = min(len(walk_sequence), target_position + current_window + 1)
        context_indices_in_window = [walk_sequence[j] for j in range(start_idx, end_idx)
                                   if target_position != j]
        return [(target_node_index, context_index) for context_index in context_indices_in_window]

    def create_batch_tensors(self, data_pairs_indices):
        current_pair_idx = 0
        while current_pair_idx < len(data_pairs_indices):
            batch_targets_list, batch_contexts_list, batch_negatives_list = [], [], []
            batch_item_counter = 0

            while batch_item_counter < self.batch_size and current_pair_idx < len(data_pairs_indices):
                target_idx, context_idx = data_pairs_indices[current_pair_idx]
                current_pair_idx += 1
                negative_candidates = self.generate_negative_samples(target_idx, context_idx)
                if len(negative_candidates) == self.num_negative:
                    batch_targets_list.append(target_idx)
                    batch_contexts_list.append(context_idx)
                    batch_negatives_list.append(negative_candidates)
                    batch_item_counter += 1

            if batch_targets_list:
                targets_tensor = torch.LongTensor(batch_targets_list)
                contexts_tensor = torch.LongTensor(batch_contexts_list).unsqueeze(1)
                negatives_tensor = torch.LongTensor(batch_negatives_list)
                yield targets_tensor, contexts_tensor, negatives_tensor

    def generate_negative_samples(self, target_idx, context_idx):
        negative_candidates = []
        attempt_limit = self.num_negative * 5
        attempts = 0
        while len(negative_candidates) < self.num_negative and attempts < attempt_limit:
            neg_node_idx = random.randint(0, self.vocab_size - 1)
            if neg_node_idx != target_idx and neg_node_idx != context_idx and neg_node_idx not in negative_candidates:
                negative_candidates.append(neg_node_idx)
            attempts += 1
        return negative_candidates


class EmbeddingTrainer:
    """Handles the Skip-gram training."""
    def __init__(self, num_vocab, embed_vector_dim, device):
        self.num_vocab = num_vocab
        self.embedding_dim = embed_vector_dim
        self.device = device
        self.model = self._build_model().to(device)
        self.optimizer = None

    def _build_model(self):
        return NodeEmbeddingModel(self.num_vocab, self.embedding_dim)

    def train(self, walks_as_indices, epochs_count, learning_rate_val, window_size, batch_size, num_negative):
        print(f"\n  Starting Node Embedding training ({epochs_count} epochs)...")
        start_train_time = time.time()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate_val)
        self.model.train()

        batch_gen = BatchGenerator(walks_as_indices, self.num_vocab, window_size, batch_size, num_negative)

        for epoch in range(epochs_count):
            epoch_stats = self._train_epoch(epoch, epochs_count, batch_gen)
            if epoch_stats['num_batches'] > 0:
                self._print_epoch_summary(epoch, epochs_count, epoch_stats)

        print("  Node Embedding training finished.")
        total_training_time = time.time() - start_train_time
        print(f"Embedding model training completed in {total_training_time:.2f} seconds.")

    def _train_epoch(self, epoch, epochs_count, batch_gen):
        total_epoch_loss = 0
        batch_process_count = 0
        epoch_start_time = time.time()

        batch_iterator = batch_gen.generate_training_batches()

        try:
            peek_batch = next(batch_iterator)
            batch_iterator = batch_gen.generate_training_batches() # Recreate iterator
            has_batches = True
        except StopIteration:
            has_batches = False

        if not has_batches:
            print(f"    Epoch {epoch+1}/{epochs_count}: No training batches generated from walks. Skipping epoch.")
            return {'num_batches': 0, 'avg_loss': 0, 'duration': 0}

        num_batches_this_epoch = 0
        for i, (targets, contexts, negatives) in enumerate(batch_iterator):
            batch_loss = self._process_batch(targets, contexts, negatives)
            total_epoch_loss += batch_loss
            batch_process_count += 1
            num_batches_this_epoch += 1
            self._print_training_progress(epoch, epochs_count, i, total_epoch_loss, batch_process_count)

        avg_epoch_loss = total_epoch_loss / batch_process_count if batch_process_count > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        return {
            'num_batches': num_batches_this_epoch,
            'avg_loss': avg_epoch_loss,
            'duration': epoch_duration
        }

    def _process_batch(self, targets, contexts, negatives):
        targets = targets.to(self.device)
        contexts = contexts.to(self.device)
        negatives = negatives.to(self.device)
        self.optimizer.zero_grad()
        loss = self.model(targets, contexts, negatives)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _print_training_progress(self, epoch, epochs_count, batch_idx, total_loss, batch_count):
        if (batch_idx + 1) % 200 == 0:
            current_avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"    Epoch {epoch+1}/{epochs_count}, Batch {batch_idx+1}, Current Avg Loss: {current_avg_loss:.4f}", end='\r')

    def _print_epoch_summary(self, epoch, epochs_count, epoch_stats):
        print(f"\n    Epoch {epoch+1}/{epochs_count} complete. "
              f"Batches processed: {epoch_stats['num_batches']}, "
              f"Average Loss: {epoch_stats['avg_loss']:.4f}, "
              f"Time: {epoch_stats['duration']:.2f}s")

    def get_node_vectors(self):
        print("  Extracting node embeddings...")
        self.model.eval()
        with torch.no_grad():
            node_vectors = self.model.node_embeddings.weight.cpu().numpy()
        print(f"Generated node embeddings shape: {node_vectors.shape}")
        return node_vectors

class ClusteringAnalyzer:
    """Performs K-Means clustering and Elbow method analysis."""
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.wcss_scores = []
        self.optimal_k_found = -1

    def find_optimal_k_elbow(self, k_candidates_range_val, num_nodes):
        print(f"\n[4/6] Finding optimal number of clusters using Elbow method (testing K={list(k_candidates_range_val)})...")
        start_time = time.time()

        max_k_possible = min(num_nodes - 1, k_candidates_range_val[-1])
        min_k_required = k_candidates_range_val.start

        if num_nodes < min_k_required:
             print(f"Error: Not enough nodes ({num_nodes}) to test the specified K range ({min_k_required} to {k_candidates_range_val[-1]}). Need at least {min_k_required} nodes.")
             self.optimal_k_found = -1
             return []

        adjusted_k_range = range(min_k_required, max_k_possible + 1)
        if len(adjusted_k_range) < 2:
             print(f"Error: Adjusted K range {list(adjusted_k_range)} contains less than 2 values. Cannot perform Elbow analysis.")
             self.optimal_k_found = -1
             return []

        print(f"  Adjusted K test range: {list(adjusted_k_range)}")

        self.wcss_scores = []
        for n_clusters in adjusted_k_range:
            try:
                if n_clusters > self.embeddings.shape[0]:
                    print(f"    Skipping K={n_clusters}: More clusters requested than data points.")
                    self.wcss_scores.append(np.nan)
                    continue

                kmeans_instance = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_instance.fit(self.embeddings)
                self.wcss_scores.append(kmeans_instance.inertia_)
                print(f"    K={n_clusters}, Within-Cluster Sum of Squares (WCSS): {kmeans_instance.inertia_:.2f}")
            except Exception as e:
                print(f"    An error occurred during K-Means for K={n_clusters}: {e}")
                self.wcss_scores.append(np.nan)

        valid_k_scores = [(k, score) for k, score in zip(adjusted_k_range, self.wcss_scores) if not np.isnan(score)]
        self.optimal_k_found = adjusted_k_range.start

        if len(valid_k_scores) >= 3:
            points_array = np.array(valid_k_scores)
            point1 = points_array[0]
            point2 = points_array[-1]

            if point2[0] == point1[0]:
                 print("Warning: Could not calculate Elbow heuristic (degenerate points). Using min K as default.")
            else:
                 A_param = point2[1] - point1[1]
                 B_param = point1[0] - point2[0]
                 C_param = point2[0]*point1[1] - point1[0]*point2[1]
                 denominator = np.sqrt(A_param**2 + B_param**2)

                 if denominator > 1e-9:
                    distances_to_line = []
                    for i in range(1, len(points_array) - 1):
                        k_val, wcss_val = points_array[i]
                        distance = abs(A_param * k_val + B_param * wcss_val + C_param) / denominator
                        distances_to_line.append(distance)

                    if distances_to_line:
                        max_dist_index = np.argmax(distances_to_line)
                        self.optimal_k_found = int(points_array[max_dist_index + 1][0])
                 else:
                     print("Warning: Could not calculate Elbow heuristic (line parameters close to zero). Using min K as default.")

        print(f"\nEstimated optimal number of clusters (N_CLUSTERS) via Elbow heuristic: {self.optimal_k_found}")
        print(f"Elbow analysis finished in {time.time() - start_time:.2f} seconds.")
        return list(adjusted_k_range)

    def perform_final_clustering(self, n_clusters):
        print(f"\n[5/6] Performing final K-Means clustering with N_CLUSTERS = {n_clusters}...")
        start_time = time.time()

        valid_n_clusters = max(1, min(n_clusters, self.embeddings.shape[0]))
        if valid_n_clusters != n_clusters:
             print(f"Warning: Adjusted final N_CLUSTERS from {n_clusters} to {valid_n_clusters} due to embedding/node count.")
        n_clusters_final = valid_n_clusters

        if n_clusters_final < 2 and self.embeddings.shape[0] >= 2:
             print(f"Warning: K-Means requires n_clusters >= 2 for more than one sample. Setting N_CLUSTERS to 2.")
             n_clusters_final = 2
        elif self.embeddings.shape[0] < 2:
             print(f"Warning: Only {self.embeddings.shape[0]} data points available. Cannot perform K-Means with N_CLUSTERS > 1.")
             n_clusters_final = 1

        try:
            final_kmeans = KMeans(n_clusters=n_clusters_final, random_state=42, n_init=10)
            cluster_labels = final_kmeans.fit_predict(self.embeddings)
            print(f"Final clustering finished in {time.time() - start_time:.2f} seconds.")
            return cluster_labels, n_clusters_final
        except Exception as e:
            print(f"Error during final K-Means clustering with K={n_clusters_final}: {e}")
            return np.zeros(self.embeddings.shape[0], dtype=int), 1

    def get_wcss_data(self):
        return self.wcss_scores

    def get_optimal_k(self):
        return self.optimal_k_found

class EmbeddingVisualizer:
    """Handles visualization using PCA and Matplotlib."""
    def __init__(self, embeddings, cluster_labels, node_id_map, num_nodes, num_edges, output_dir="plots"):
        self.embeddings = embeddings
        self.cluster_labels = cluster_labels
        self.node_id_map = node_id_map
        if cluster_labels is not None and len(cluster_labels) > 0:
             self.num_clusters = len(np.unique(cluster_labels))
        else:
             self.num_clusters = 0

        self.num_graph_nodes = num_nodes
        self.num_graph_edges = num_edges
        self.pca_embeddings_2d = None
        self.explained_variance_ratio = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Create output directory if it doesn't exist

    def reduce_dimensions_pca(self, n_components_val=2):
        if self.embeddings is None or self.embeddings.shape[0] < n_components_val:
            print(f"Skipping PCA: Not enough data points ({self.embeddings.shape[0]}) for {n_components_val} components.")
            self.pca_embeddings_2d = None
            self.explained_variance_ratio = None
            return

        print(f"\n[6/6] Visualizing clusters using PCA with {n_components_val} components...")
        start_time = time.time()
        try:
            pca_transformer = PCA(n_components=n_components_val)
            self.pca_embeddings_2d = pca_transformer.fit_transform(self.embeddings)
            self.explained_variance_ratio = pca_transformer.explained_variance_ratio_
            total_explained_variance = self.explained_variance_ratio.sum() * 100
            print(f"  PCA reduction complete ({n_components_val} components). Explained variance: {total_explained_variance:.2f}%")
        except Exception as e:
             print(f"Error during PCA reduction: {e}")
             self.pca_embeddings_2d = None
             self.explained_variance_ratio = None
        print(f"Visualization PCA setup finished in {time.time() - start_time:.2f} seconds.")

    def plot_and_save_clustered_embeddings(self, sample_labels_count=0, filename="clustered_embeddings_pca.png"):
        """Plots the 2D PCA embeddings colored by cluster and saves it to a file."""
        if self.pca_embeddings_2d is None or (self.cluster_labels is None and self.num_graph_nodes > 1):
            print("Skipping cluster plot: PCA reduction failed or cluster labels missing/insufficient nodes.")
            return
        if self.pca_embeddings_2d.shape[0] == 0:
             print("Skipping cluster plot: No embeddings to plot.")
             return
        if self.cluster_labels is not None and self.cluster_labels.shape[0] != self.pca_embeddings_2d.shape[0]:
             print("Skipping cluster plot: Mismatch between embedding count and label count.")
             return

        print(f"  Plotting and saving clustered embeddings to {os.path.join(self.output_dir, filename)}...")
        plt.figure(figsize=(12, 10))

        if self.cluster_labels is not None and self.num_clusters > 1:
             scatter_plot = plt.scatter(self.pca_embeddings_2d[:, 0], self.pca_embeddings_2d[:, 1],
                                    c=self.cluster_labels, cmap='viridis', alpha=0.7, s=50)
        else:
             # Plot without cluster colors if only 0 or 1 cluster found
             scatter_plot = plt.scatter(self.pca_embeddings_2d[:, 0], self.pca_embeddings_2d[:, 1],
                                    color='grey', alpha=0.7, s=50)


        if sample_labels_count > 0 and self.node_id_map is not None and self.num_graph_nodes > 0:
            num_samples = min(sample_labels_count, self.num_graph_nodes)
            if num_samples > 0:
                indices_to_label = random.sample(range(self.num_graph_nodes), num_samples)
                for i in indices_to_label:
                    node_identifier = self.node_id_map.get(i, f"Unknown_{i}")
                    plt.text(self.pca_embeddings_2d[i, 0] + 0.05, self.pca_embeddings_2d[i, 1] + 0.05,
                            str(node_identifier), fontsize=8, alpha=0.8)

        pca1_variance = self.explained_variance_ratio[0] * 100 if self.explained_variance_ratio is not None and len(self.explained_variance_ratio) > 0 else 0
        pca2_variance = self.explained_variance_ratio[1] * 100 if self.explained_variance_ratio is not None and len(self.explained_variance_ratio) > 1 else 0

        plt.title(f'Node Embeddings Clustered (K={self.num_clusters}) using PCA\n(Nodes: {self.num_graph_nodes}, Edges: {self.num_graph_edges})')
        plt.xlabel(f'PCA Component 1 ({pca1_variance:.1f}%)')
        plt.ylabel(f'PCA Component 2 ({pca2_variance:.1f}%)')

        if self.cluster_labels is not None and self.num_clusters > 1:
            cluster_legend_labels = [f'Cluster {i}' for i in range(self.num_clusters)]
            try:
                legend_handles, _ = scatter_plot.legend_elements()
                if len(legend_handles) == self.num_clusters:
                     plt.legend(handles=legend_handles, labels=cluster_legend_labels, title="Clusters")
                else:
                     print("Warning: Legend elements do not match cluster count. Using colorbar.")
                     plt.colorbar(scatter_plot, label='Cluster ID')
            except Exception as e:
                 print(f"Warning: Error creating legend: {e}. Using colorbar.")
                 plt.colorbar(scatter_plot, label='Cluster ID')

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(self.output_dir, filename))
            print(f"  Clustered embeddings plot saved successfully.")
        except Exception as e:
             print(f"Error saving clustered embeddings plot: {e}")
        plt.close() # Close the plot after saving


# --- Main Execution Pipeline ---

if __name__ == "__main__":
    elbow_k_range = range(2, 7)
    plot_output_directory = "clustering_plots" # Define output directory for plots

    execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {execution_device}")

    # Step 1: Load Data
    data_processor = DataProcessor("payments.xlsx")
    data_df = data_processor.load_and_process()
    if data_df is None:
         exit()

    # Step 2: Build Graph
    graph_creator = GraphCreator()
    graph, nodes, node_id_to_index_map, index_to_node_id_map, num_graph_nodes, graph_ok = graph_creator.build_and_aggregate_graph(data_df, elbow_k_range)

    if not graph_ok:
        exit()

    if graph.number_of_edges() == 0 and num_graph_nodes > 1:
        print("Graph has nodes but no edges. Node2Vec will not work. Exiting.")
        exit()
    elif graph.number_of_edges() == 0 and num_graph_nodes <= 1:
        print("Graph has no edges or only one node. Cannot perform Node2Vec. Exiting.")
        exit()

    # Step 3: Generate Walks and Train Embedding Model
    print("\n[3/6] Generating biased random walks and training embedding model...")

    # Use the refactored walk generation logic
    generated_walks_ids = generate_random_walks(
        graph, 15, 40, nodes, node_id_to_index_map, 1.0, 1.0
    )

    if not generated_walks_ids:
         print("Error: No valid walks generated. Cannot train embedding model. Exiting.")
         exit()

    generated_walks_indices = convert_walks_to_indices(generated_walks_ids, node_id_to_index_map)

    embedding_trainer = EmbeddingTrainer(
        num_vocab=num_graph_nodes,
        embed_vector_dim=64,
        device=execution_device
    )
    embedding_trainer.train(
        generated_walks_indices,
        epochs_count=5,
        learning_rate_val=0.01,
        window_size=5,
        batch_size=128,
        num_negative=5
    )
    node_embeddings = embedding_trainer.get_node_vectors()

    if node_embeddings is None or node_embeddings.shape[0] == 0:
        print("Error: No node embeddings generated. Cannot perform clustering or visualization. Exiting.")
        exit()
    if node_embeddings.shape[0] < 2:
         print(f"Only {node_embeddings.shape[0]} embedding(s) generated. Cannot perform clustering with K > 1. Skipping Elbow and multi-cluster plot.")
         optimal_cluster_k = 1
         final_cluster_labels = np.zeros(node_embeddings.shape[0], dtype=int)
         final_num_clusters = 1
         tested_k_values = []
         wcss_values_for_plot = []
    else:
        # Step 4: Find Optimal K using Elbow Method
        clustering_analyzer = ClusteringAnalyzer(node_embeddings)
        tested_k_values = clustering_analyzer.find_optimal_k_elbow(
            elbow_k_range, num_graph_nodes
        )
        optimal_cluster_k = clustering_analyzer.get_optimal_k()
        wcss_values_for_plot = clustering_analyzer.get_wcss_data()

        # Step 5: Perform Final Clustering with Optimal K
        if optimal_cluster_k > 0 and optimal_cluster_k <= num_graph_nodes:
            final_cluster_labels, final_num_clusters = clustering_analyzer.perform_final_clustering(optimal_cluster_k)
            if final_cluster_labels is not None:
                node_id_clusters = {index_to_node_id_map[i]: cluster for i, cluster in enumerate(final_cluster_labels)}
            else:
                 node_id_clusters = {}
                 final_num_clusters = 0
        else:
            print("\nOptimal K could not be determined or is invalid. Skipping final clustering.")
            final_cluster_labels = None
            final_num_clusters = 0


    # Step 6: Visualize Clustered Embeddings with PCA (save to file)
    # Only plot if we have embeddings AND enough for PCA (>1 point for 2D PCA)
    if node_embeddings is not None and node_embeddings.shape[0] >= 2:
        sample_labels_count_val = 0 # Set to > 0 to show some node labels

        visualization_plotter_pca = EmbeddingVisualizer(
            embeddings=node_embeddings,
            cluster_labels=final_cluster_labels, # Can be None if clustering skipped/failed
            node_id_map=index_to_node_id_map,
            num_nodes=num_graph_nodes,
            num_edges=graph.number_of_edges(),
            output_dir=plot_output_directory
        )
        # Pass n_components directly
        visualization_plotter_pca.reduce_dimensions_pca(n_components_val=2)

        if visualization_plotter_pca.pca_embeddings_2d is not None:
             # If clustering resulted in 0 or 1 cluster, or was skipped, plot without cluster colors
             if final_cluster_labels is not None and final_num_clusters > 1:
                visualization_plotter_pca.plot_and_save_clustered_embeddings(sample_labels_count_val)
             else:
                 print("\nPlotting embeddings with PCA, but without cluster colors (0 or 1 cluster).")
                 # Temporarily set cluster_labels to None in plotter to force grey/single color plot
                 temp_labels = visualization_plotter_pca.cluster_labels
                 visualization_plotter_pca.cluster_labels = None
                 visualization_plotter_pca.num_clusters = 1 # Represent as single group
                 visualization_plotter_pca.plot_and_save_clustered_embeddings(sample_labels_count_val, filename="embeddings_pca_no_clustering.png")
                 # Restore original labels if needed later (not needed in this script)
                 visualization_plotter_pca.cluster_labels = temp_labels
                 visualization_plotter_pca.num_clusters = len(np.unique(temp_labels)) if temp_labels is not None else 0


        else:
             print("\nSkipping PCA visualization due to insufficient data or PCA failure.")

    else:
         print("\nSkipping PCA visualization due to insufficient embeddings (< 2 points).")


    print("\n--- Assignment Complete ---")
    print(f"Plots saved to the '{plot_output_directory}' directory.")