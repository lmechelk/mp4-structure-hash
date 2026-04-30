"""
MP4 COMPARE 2.1
Lucie Mechelk, 2026

Description:
A comprehensive tool for analyzing MP4 structural similarities

Features:
1. '11' (1-to-1): Directly parses and compares two MP4 files using their deep tree structure.
2. '1n' (1-to-N): Evaluates a batch dataset (JSON/CSV) against a reference video, generating ROC curves & exporting raw scores.
3. 'nn' (N-to-N): Computes a full similarity matrix across a batch dataset for clustering.
"""

import json
import csv
import sys
import os
import argparse

# Data processing and visualization
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Import the parser for live mode
from mp4_parse import Mp4Parser 


# ==========================================
# FEATURE EXTRACTION & MATH
# ==========================================

class Mp4Digest:
    """
    Transforms MP4 data (either deep tree or flat signature) into an n-gram feature set
    
    input: filename, tree (dict), n (gram size)
    output: n-gram feature set
    """
    def __init__(self, filename, n, signature=None, tree=None):
        self.filename = filename
        self.n = n # n-gram size (e.g., 2 for bigrams, 3 for trigrams)
        
        # Use the true tree structure if available, otherwise fallback to the string signature
        if tree and len(tree) > 0:
            # TODO: doestn work???
            self.atom_sequence = self._flatten_tree(tree)
        elif signature:
            self.atom_sequence = self._tokenize(signature)
            # print(f"DEBUG: Extracted {len(self.atom_sequence)} atoms from signature.")
        else:
            self.atom_sequence = []
            
        self.features = self._generate_ngrams(self.atom_sequence, self.n)

    def _flatten_tree(self, nodes):
        # TODO: doestn work???
        """
        Depth-First Search (DFS) to extract a linear sequence of atoms from a nested tree.
        Example: {"type": "moov", "children": [{"type": "mvhd"}]} -> ["moov", "mvhd"]
        """
        sequence = [] # Linear sequence
        for node in nodes:
            sequence.append(node["type"])
            if "children" in node:
                sequence.extend(self._flatten_tree(node["children"]))
        print(f"DEBUG: Extracted {len(sequence)} atoms from tree.")
        return sequence

    def _tokenize(self, signature):
        """
        Fallback: Chunks a flat signature string into 4-character ISO box names.
        Example: "ftypmoov" -> ["ftyp", "moov"]
        
        Input: Raw string representation of MP4 structure
        Output: A list of atoms in the order they appear in the structure
        """
        clean_str = signature.replace('|', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
        # print(f"DEBUG: Extracted {len(clean_str)//4} atoms from signature.")
        return [clean_str[i:i+4] for i in range(0, len(clean_str), 4)]

    def _generate_ngrams(self, sequence, n):
        """
        Generates distinct, sorted n-grams from the linear atom sequence.
        
        Input: Sequence of atoms in the order they appear in the structure
        Output: A sorted list of unique n-grams as tuples
        """
        
        # Fallback for very small files
        if len(sequence) < n:
            return [tuple(sequence)] if sequence else []
        
        # Slide a window of size 'n' across the sequence of atoms
        ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)] 
        
        # Convert to a set to remove duplicates, then back to a sorted list
        return sorted(list(set(ngrams)))

    def get_features(self):
        """Returns the generated n-grams as a mathematical Set for similarity calculation.
        
        Input: Sequence of atoms in the order they appear in the structure
        Output: A mathematical Set of unique n-grams as tuples
        """
        return set(self.features)


class SimilarityCalculator:
    """Calculates Jaccard and Tversky indices between two feature sets.
    
    Input: features_a, features_b, alpha (weight for False Negatives), beta (weight for False Positives)
    Output: Jaccard and Tversky similarity scores
    """
    def __init__(self, features_a, features_b, alpha=0.0, beta=1.0):
        self.set_a = features_a
        self.set_b = features_b
        self.alpha = alpha # Weight: elements in A but missing in B
        self.beta = beta   # Weight: new elements in B that are not in A

    def calculate_jaccard(self):
        """Calculates the standard Jaccard similarity (Intersection / Union)."""
        
        # Both files are empty
        if not self.set_a and not self.set_b: 
            print("Both files are empty")
            return 1.0 
        
        # Extract the features and convert them to Python Sets
        intersection = len(self.set_a.intersection(self.set_b))         # True Positives
        union = len(self.set_a.union(self.set_b))                       # True Positives + True Negatives
        return float(intersection) / float(union) if union > 0 else 0.0 # True Positives / (True Positives + True Negatives)

    def calculate_tversky(self):
        """Calculates the asymmetrical Tversky similarity index."""
        
        # TODO: auch wenn 1 leer
        # Both files are empty
        if not self.set_a and not self.set_b: 
            # TODO: hier exception oder none
            print("Both files are empty")
            return 1.0 # auf keinen Fall 1.0 zurück!!!!
        
        # Extract the features and convert them to Python Sets
        # TODO: prototyp / variante
        intersection = len(self.set_a.intersection(self.set_b)) # True Positives
        diff_a = len(self.set_a.difference(self.set_b))         # False Negatives
        diff_b = len(self.set_b.difference(self.set_a))         # False Positives
        
        # Denominator applies the alpha and beta weights to the differences
        denominator = intersection + (self.alpha * diff_a) + (self.beta * diff_b)   # True Positives + False Negatives + False Positives
        return float(intersection) / float(denominator) if denominator > 0 else 0.0 # True Positives / (True Positives + False Negatives + False Positives)


# ==========================================
# DATA LOADING & HELPER
# ==========================================

def load_dataset(file_path):
    """Loads the parsed dataset from a JSON or CSV file."""
    print(f"DEBUG: Attempting to load file: {file_path}")
    dataset = []
    
    if not os.path.isfile(file_path):
        print(f"ERROR: File does not exist at path: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # 1. Handle JSON files
        if file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        
        # 2. Handle CSV (Excel) files
        elif file_path.lower().endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';') 
                
                for row in reader:
                    status = row.get("Status", "N/A").lower()
                    
                    # Only process files that were parsed successfully
                    if status == "success" or "success" in status:
                        tree_data = json.loads(row.get("Tree_JSON", "[]")) if row.get("Tree_JSON") else []
                        dataset.append({
                            "filename": row.get("Filename", row.get("filename", "")),
                            "signature": row.get("Signature", row.get("signature", "")),
                            "tree": tree_data
                        })
        
        if not dataset:
            print("WARNING: Dataset was loaded, but it is EMPTY (wrong column names or no 'success' status?)", file=sys.stderr)
        else:
            print(f"SUCCESS: Loaded {len(dataset)} videos into memory.")
            
    except Exception as e:
        print(f"CRITICAL ERROR during loading: {e}", file=sys.stderr)
        sys.exit(1)
        
    return dataset

def load_ground_truth(file_path):
    """Loads an optional external Ground Truth mapping (Format: Filename;Label)."""
    gt_map = {}
    if not file_path or not os.path.isfile(file_path):
        return gt_map
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter=';'):
                if len(row) >= 2: gt_map[row[0]] = int(row[1])
        print(f"SUCCESS: Loaded {len(gt_map)} labels from Ground Truth CSV.")
    except Exception as e:
        print(f"ERROR loading Ground Truth file: {e}", file=sys.stderr)
    return gt_map

def get_category(filename, gt_map=None):
    """
    Determines the category of a file based strictly on its root folder.
    Example: 'WhatsApp/flat/video.mp4' -> 'whatsapp'
    Allows an optional override via a ground-truth CSV map.
    """
    # 1. Priority: Ground Truth Map (if provided)
    if gt_map and filename in gt_map:
        return "native" if gt_map[filename] == 0 else "manipulated"
    
    # 2. Strict splitting at the first folder (Root directory name)
    clean_path = filename.replace('\\', '/')
    if '/' in clean_path:
        return clean_path.split('/')[0].lower() # e.g., 'd03_huawei_p9' or 'imagine'
    
    return "root" # Fallback if the file is not in any folder


# ==========================================
# EVALUATION MODES
# ==========================================

def run_one_to_one_live(file1, file2, n_gram_size, metric, alpha, beta):
    """Live Mode: Parses and compares two actual MP4 files directly from the disk."""
    print(f"\nParsing files directly...")
    
    # Call the external parser script logic
    res1 = Mp4Parser(file1).parse()
    res2 = Mp4Parser(file2).parse()

    # Check for parsing errors
    if res1["status"] != "success": sys.exit(f"ERROR parsing {file1}: {res1.get('error_message')}")
    if res2["status"] != "success": sys.exit(f"ERROR parsing {file2}: {res2.get('error_message')}")

    print(f"\n--- FILE 1: {res1['filename']} ---\n{res1['signature']}")
    print(f"\n--- FILE 2: {res2['filename']} ---\n{res2['signature']}")

    # Extract features using the structural tree
    digest1 = Mp4Digest(res1['filename'], n_gram_size, tree=res1['tree'])
    digest2 = Mp4Digest(res2['filename'], n_gram_size, tree=res2['tree'])

    # Calculate similarity
    calc = SimilarityCalculator(digest1.get_features(), digest2.get_features(), alpha, beta)
    score = calc.calculate_jaccard() if metric == "j" else calc.calculate_tversky()

    metric_name = "JACCARD" if metric == "j" else "TVERSKY"

    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Metric: {metric_name} (n={n_gram_size}) | Score: {score:.4f}")
    print("==========================\n")


def run_one_to_n(dataset, ref_filename, n_gram_size, metric, alpha, beta, plot, gt_map=None, scores_out="output_1n.csv"):
    """
    1-to-N Evaluation Mode (ROC generation):
    Class 0: Files originating from the SAME folder as the reference (True Negatives).
    Class 1: Files originating from a DIFFERENT folder than the reference (True Positives).
    """
    print(f"\n--- 1-to-N EVALUATION (Reference-based) ---")
    
    # Determine the class 0 (Baseline) category
    ref_category = get_category(ref_filename, gt_map)
    print(f"Reference Category (Class 0): {ref_category}")
    
    # Find the reference item
    ref_item = next((i for i in dataset if i["filename"] == ref_filename), None)
    if not ref_item: sys.exit(f"ERROR: Reference '{ref_filename}' not found in the dataset.")
    
    # Calculate features for the reference file
    ref_features = Mp4Digest(ref_filename, n_gram_size, tree=ref_item.get("tree"), signature=ref_item.get("signature")).get_features()

    # Choose the evaluation metric
    metric_name = "Jaccard" if metric == "j" else "Tversky"
    
    # List to store raw results for CSV
    raw_results = []                         

    # Pass 1: Collect ALL files and calculate their similarity to the reference
    for video in dataset:
        cat = get_category(video["filename"], gt_map)
        
        test_features = Mp4Digest(video["filename"], n_gram_size, tree=video.get("tree"), signature=video.get("signature")).get_features()
        calc = SimilarityCalculator(ref_features, test_features, alpha, beta)
        
        sim_score = calc.calculate_jaccard() if metric == "j" else calc.calculate_tversky()
        dist_score = 1.0 - sim_score # Invert similarity to get distance/anomaly score for ROC
        
        # Determine Ground Truth: 0 if same folder (baseline), 1 if different folder (anomaly/fake)
        is_anomaly = 0 if cat == ref_category else 1
        
        raw_results.append({
            "Filename": video["filename"],
            "Category": cat.upper(),
            "Ground_Truth": is_anomaly,
            "Similarity_Score": round(sim_score, 4),
            "Anomaly_Distance": round(dist_score, 4)
        })

    # Export raw results to CSV
    if scores_out:
        try:
            with open(scores_out, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["Filename", "Category", "Ground_Truth", "Similarity_Score", "Anomaly_Distance"], delimiter=';')
                writer.writeheader()
                writer.writerows(raw_results)
            print(f"SUCCESS: Raw evaluation scores exported to '{scores_out}'")
        except Exception as e:
            print(f"ERROR: Could not write scores to {scores_out}: {e}")

    # Pass 2: Prepare stats dictionary for plotting based on raw_results
    stats = {}
    baseline_scores = [r["Anomaly_Distance"] for r in raw_results if r["Ground_Truth"] == 0]

    for r in raw_results:
        if r["Ground_Truth"] == 1:
            cat = r["Category"]
            if cat not in stats:
                stats[cat] = {"true": [], "scores": []}
            stats[cat]["true"].append(1) # Ground Truth: Different folder (Class 1)
            stats[cat]["scores"].append(r["Anomaly_Distance"])

    # Append baseline (Class 0) to EVERY comparison group
    # This ensures that each ROC curve has a common 'native' baseline to compare against.
    for cat in stats:
        for b_score in baseline_scores:
            stats[cat]["true"].append(0) # Ground Truth: Same folder (Class 0)
            stats[cat]["scores"].append(b_score)

    # Output / Plotting
    if plot and stats:
        
        plt.figure(figsize=(15, 6))
        plt.subplots_adjust(right=0.7) 
        
        # Plot ROC curves   
        for cat, data in stats.items():
            if len(set(data["true"])) > 1: # Requires both Class 0 and Class 1 to plot an ROC curve
                fpr, tpr, _ = roc_curve(data["true"], data["scores"]) # Compute ROC curve
                plt.plot(fpr, tpr, lw=2, label=f"{cat.upper()} (AUC={auc(fpr, tpr):.2f})") # Plot ROC curve
        
        # Plot baseline
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        if metric == "t":
            metric_info = f"Tversky \u03b1={alpha} \u03b2={beta}"
        else:
            metric_info = "Jaccard"
        plt.title(f"1-to-N Evaluation ({metric_info}, n={n_gram_size})\nRef: {ref_filename}")
        plt.xlabel(f"False Positive Rate (Same Category: {ref_category})")
        plt.ylabel("True Positive Rate (Different Category)")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        
        # Save plot
        plt.savefig('output.png', bbox_inches='tight')
        print("SUCCESS: Plot saved as 'output.png'")
        
        # Show plot
        plt.show()
        
    elif not stats:
        print("WARNING: No comparison categories found, check your paths in the CSV.")

def run_n_to_n(dataset, n_gram_size, metric, alpha, beta, output_csv):
    """Batch Matrix: Computes pairwise scores across the entire dataset for clustering."""
    print(f"\n--- N-to-N EVALUATION (Similarity Matrix) ---")
    
    # Pre-calculate features to optimize execution time
    digests = {v["filename"]: Mp4Digest(v["filename"], n_gram_size, tree=v.get("tree"), signature=v.get("signature")).get_features() for v in dataset}
    filenames = list(digests.keys())
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([""] + filenames) # Write header row
        
        for f_a in filenames:
            row = [f_a]
            for f_b in filenames:
                c = SimilarityCalculator(digests[f_a], digests[f_b], alpha, beta)
                score = c.calculate_jaccard() if metric == 'j' else c.calculate_tversky()
                row.append(f"{score:.4f}")
            writer.writerow(row)
    print(f"SUCCESS: Matrix saved to {output_csv}")


# ==========================================
# CLI ENTRY POINT & STEERING
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP4 Structure Comparison & Evaluation")
    parser.add_argument("--mode", choices=["11", "1n", "nn"], required=True, help="Operating mode")
    
    # Args for 1-to-1 live mode
    parser.add_argument("--f1", type=str, help="First MP4 file (for 11 mode)")
    parser.add_argument("--f2", type=str, help="Second MP4 file (for 11 mode)")
    
    # Args for batch modes (1-to-N, N-to-N)
    parser.add_argument("--dataset", type=str, help="Path to parsed JSON or CSV dataset")
    parser.add_argument("--ref", type=str, help="Reference filename in dataset (for 1n mode)")
    parser.add_argument("--gt-file", type=str, help="Optional external ground truth CSV (Format: filename;label)")
    parser.add_argument("--plot", action="store_true", help="Show ROC plot (for 1n mode)")
    parser.add_argument("--scores-out", type=str, default="output_1n.csv", help="Output file for raw 1n scores") # NEU
    parser.add_argument("--matrix-out", type=str, default="matrix.csv", help="Output file name (for nn mode)")
    
    # Algorithmic parameters
    parser.add_argument("--ngram", "-n", type=int, default=2, help="N-Gram size (e.g., 2, 3, 4)")
    parser.add_argument("--metric", "-m", choices=["j", "t"], default="t", help="Similarity metric: 'j' (Jaccard) or 't' (Tversky)")
    parser.add_argument("--alpha", "-a", type=float, default=0.0, help="Tversky Alpha (Weight for missing features)")
    parser.add_argument("--beta", "-b", type=float, default=1.0, help="Tversky Beta (Weight for added features)")

    args = parser.parse_args()

    if args.mode == "11":
        if not args.f1 or not args.f2:
            sys.exit("ERROR: --f1 and --f2 are required for 1-to-1 mode ('11').")
        run_one_to_one_live(args.f1, args.f2, args.ngram, args.metric, args.alpha, args.beta)
        
    elif args.mode in ["1n", "nn"]:
        if not args.dataset:
            sys.exit("ERROR: --dataset (.json or .csv) is required for batch modes.")
            
        dataset = load_dataset(args.dataset)
        gt_map = load_ground_truth(args.gt_file)
        
        if args.mode == "1n":
            if not args.ref: sys.exit("ERROR: --ref is required for 1-to-N mode ('1n').")
            # Aufruf aktualisiert, übergibt nun args.scores_out
            run_one_to_n(dataset, args.ref, args.ngram, args.metric, args.alpha, args.beta, args.plot, gt_map, args.scores_out)
            
        elif args.mode == "nn":
            run_n_to_n(dataset, args.ngram, args.metric, args.alpha, args.beta, args.matrix_out)