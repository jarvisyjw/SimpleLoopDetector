from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import argparse
import numpy as np

def load_scores(file_path):
    """
    Load precision-recall data from a file.
    
    Args:
        file_path (str): Path to the file containing precision-recall data.
        
    Returns:
        list: A list of tuples containing precision and recall values.
    """
    scores = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                _, _, label, score = line.strip().split()
                scores.append(float(score))
                labels.append(int(label))
    return scores, labels

def max_recall(precision: np.ndarray, recall: np.ndarray):
    idx = np.where(precision == 1.0)
    max_recall = np.max(recall[idx])
    return max_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Precision-Recall Curve")
    parser.add_argument("file_path", type=str, help="Path to the file containing precision-recall data")
    args = parser.parse_args()
    scores, labels = load_scores(args.file_path)
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = average_precision_score(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Average Precision = {average_precision*100:.5f}, Max Recall = {max_recall(precision, recall)*100:.5f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()
    print(f"AP {average_precision:.8f}, Max Recall {max_recall(precision, recall):.8f}")