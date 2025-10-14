import os
import time
import pickle
from sklearn.datasets import load_svmlight_file
import scipy.sparse as sps

def pickle_dataset(input_path, output_path):
    """
    Load dataset in LIBSVM format and save it as a pickle file.
    
    Parameters
    ----------
    input_path : str
        Path to the input LIBSVM dataset (.bz2 file).
    output_path : str
        Path to save the output pickle file.
    """
    start = time.time()
    print(f"Loading dataset from: {input_path}")

    # Load data
    A, y = load_svmlight_file(input_path)

    # Convert to dense if sparse
    if sps.issparse(A):
        A = A.toarray()

    # Save as pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump((A, y), f, protocol=4)

    elapsed = time.time() - start
    print(f"Dataset saved to {output_path} (shape={A.shape}, time={elapsed:.2f}s)")


if __name__ == "__main__":
    input_path = os.path.expanduser("../data/epsilon_normalized.bz2")
    output_path = os.path.expanduser("../data/epsilon.pickle")
    pickle_dataset(input_path, output_path)
