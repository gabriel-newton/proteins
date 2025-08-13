import pandas as pd
import numpy as np
import gc
import json
import os
import sys

# Import the Node and BinarySearchTree classes from the previous block
# Assuming they are in the same script or a local file
class Node:
    """A node in the Binary Search Tree for 4D angle keys."""
    def __init__(self, key):
        self.key = key
        self.count = 1
        self.left = None
        self.right = None

class BinarySearchTree:
    """A Binary Search Tree to store and count occurrences of 4D angle combinations."""
    def __init__(self):
        self.root = None
        self.total_nodes = 0
        self.highest_peak_count = 0
        self.highest_peak_key = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
            self.total_nodes = 1
            self.highest_peak_count = 1
            self.highest_peak_key = key
        else:
            self._insert_recursive(self.root, key)

    def _insert_recursive(self, current_node, key):
        if key == current_node.key:
            current_node.count += 1
            if current_node.count > self.highest_peak_count:
                self.highest_peak_count = current_node.count
                self.highest_peak_key = current_node.key
            return
        elif key < current_node.key:
            if current_node.left is None:
                current_node.left = Node(key)
                self.total_nodes += 1
            else:
                self._insert_recursive(current_node.left, key)
        else: # key > current_node.key
            if current_node.right is None:
                current_node.right = Node(key)
                self.total_nodes += 1
            else:
                self._insert_recursive(current_node.right, key)

    def find_highest_peak(self):
        return self.highest_peak_key, self.highest_peak_count

    def get_all_nodes(self):
        nodes = []
        self._get_all_nodes_recursive(self.root, nodes)
        return nodes

    def _get_all_nodes_recursive(self, current_node, nodes):
        if current_node:
            nodes.append((current_node.key, current_node.count))
            self._get_all_nodes_recursive(current_node.left, nodes)
            self._get_all_nodes_recursive(current_node.right, nodes)


# --- MAIN SCRIPT ---
FILE_PATH = '2mer_angles_8d.csv'
OUTPUT_JSON_FILE = 'bst_top_peaks.json'
CHUNK_SIZE = 1000000  # Process 1 million rows at a time

def process_file_with_bst(file_path):
    """
    Reads a large CSV file in chunks and uses a BST to count 4D angle occurrences.
    """
    bst = BinarySearchTree()
    
    print(f"Starting to process file: {file_path}")
    
    # Use pandas read_csv with an iterator to process the file in chunks
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE)):
        print(f"Processing chunk {i+1}...")

        # Convert cos/sin pairs to degrees using numpy.arctan2
        # Note: arctan2 returns radians, so we convert to degrees
        # and round to the nearest integer
        chunk['phi_i'] = np.round(np.rad2deg(np.arctan2(chunk['phi_i_sin'], chunk['phi_i_cos']))).astype(int)
        chunk['psi_i'] = np.round(np.rad2deg(np.arctan2(chunk['psi_i_sin'], chunk['psi_i_cos']))).astype(int)
        chunk['phi_i+1'] = np.round(np.rad2deg(np.arctan2(chunk['phi_i+1_sin'], chunk['phi_i+1_cos']))).astype(int)
        chunk['psi_i+1'] = np.round(np.rad2deg(np.arctan2(chunk['psi_i+1_sin'], chunk['psi_i+1_cos']))).astype(int)
        
        # Iterate over the rows of the chunk and insert into the BST
        for _, row in chunk.iterrows():
            key = (row['phi_i'], row['psi_i'], row['phi_i+1'], row['psi_i+1'])
            bst.insert(key)
        
        # Explicitly delete chunk to free up memory
        del chunk
        gc.collect()

    print("\nProcessing complete.")
    peak_key, peak_count = bst.find_highest_peak()
    print(f"Total unique angle combinations found: {bst.total_nodes}")
    print(f"Highest peak found at key {peak_key} with a count of {peak_count}.")
    
    return bst


if __name__ == "__main__":
    try:
        # Create a dummy file for demonstration since the actual file is not available
        if not os.path.exists(FILE_PATH):
            print("Creating a dummy CSV file for demonstration...")
            num_rows = 1_000_000
            data = pd.DataFrame({
                'phi_i_cos': np.random.uniform(-1, 1, num_rows),
                'phi_i_sin': np.random.uniform(-1, 1, num_rows),
                'psi_i_cos': np.random.uniform(-1, 1, num_rows),
                'psi_i_sin': np.random.uniform(-1, 1, num_rows),
                'phi_i+1_cos': np.random.uniform(-1, 1, num_rows),
                'phi_i+1_sin': np.random.uniform(-1, 1, num_rows),
                'psi_i+1_cos': np.random.uniform(-1, 1, num_rows),
                'psi_i+1_sin': np.random.uniform(-1, 1, num_rows),
            })
            data.to_csv(FILE_PATH, index=False)
            print(f"Dummy file '{FILE_PATH}' created with {num_rows} rows.")

        # Run the main function
        final_bst = process_file_with_bst(FILE_PATH)
        
        # Get all nodes, sort by count, and take the top 10
        all_nodes = final_bst.get_all_nodes()
        sorted_nodes = sorted(all_nodes, key=lambda x: x[1], reverse=True)
        top_10_peaks = sorted_nodes[:10]

        # Format the data for JSON output
        json_output = []
        for key, count in top_10_peaks:
            json_output.append({
                "phi_i": key[0],
                "psi_i": key[1],
                "phi_i+1": key[2],
                "psi_i+1": key[3],
                "count": count
            })

        # Write the JSON output to a file
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(json_output, f, indent=4)
        
        print(f"\nSuccessfully wrote the top 10 peaks to '{OUTPUT_JSON_FILE}'.")
            
    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
        sys.exit(1)
