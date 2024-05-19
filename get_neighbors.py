def extract_neighbors(edge_index, patch_num):
    """
    Extract neighbors from edge_index for a given patch number.
    
    Parameters:
    edge_index (numpy.ndarray): The edge index array.
    patch_num (int): The patch number to extract neighbors from.
    
    Returns:
    neighbors (list): The list of neighbors for the given patch number.
    """
    
    # Ensure the edge_index array is in the expected shape
    assert edge_index.ndim == 4, "edge_index should be a 4D array"    
    c_neighbors = edge_index[0, 0, patch_num, :]
    neighbors = c_neighbors.numpy().squeeze().tolist()
    
    return neighbors
