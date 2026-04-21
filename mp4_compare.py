"""
MP4 COMPARE

Description:
Transforms a parsed MP4 structural string into a mathematical feature set 
(Similarity Digest) for Approximate Matching. The algorithm extracts 
atom-based n-grams (sequences of 4-character ISO box names) to prepare 
the topological data for set-based similarity comparisons using the 
Jaccard coefficient and the asymmetrical Tversky index.

Acknowledgments & Credits:
1. Theoretical Framework:
   The n-gram tokenization and feature set generation implements the 
   theoretical forensic concepts proposed by Klier & Baier 2025 
   (Mp4 Structure Similarity Hash).
   https://github.com/SamKlier/mssh
"""

import os # For file handling and path operations


# Defines the lengths of the sliding window for feature extraction
# Default is 2 (Bigrams), 3 (Trigrams), and 4 (Quadgrams)
NGRAM_SIZES = (2, 3, 4)

# Tversky Parameters
# Alpha heavily weights False Negatives (Features missing in B)
# Beta heavily weights False Positives (New features added in B)
# Note: If Alpha=1.0 and Beta=1.0, Tversky equals Jaccard mathematically.
TVERSKY_ALPHA = 1.0
TVERSKY_BETA = 0.0  # Asymmetric Containment configuration


class Mp4StructureDigest:
    """
    Transforms a parsed MP4 structural string into a mathematical feature 
    set (Similarity Digest). It extracts atom-based n-grams (sequences 
    of 4-character box names) to prepare the data for set-based similarity 
    comparisons like Jaccard or Tversky.
    """
    
    def __init__(self, filename, raw_string):
        """
        Initializes the digest generator
        
        input: filename: The name of the MP4 file (for metadata purposes)
               raw_string: Raw parsed string representation of MP4 structure
        output: None (the digest is generated and stored internally)
        """
        self.filename = os.path.basename(filename)
        self.raw_string = raw_string
        
        # Step 1: Tokenize the raw string into an ordered list of atoms
        self.atom_sequence = self._tokenize(raw_string)
        
        # Step 2: Build the final dictionary containing the features
        self.digest = self._build_digest()

    def _tokenize(self, raw_string):
        """
        Cleans the string of all syntactical separators and converts it 
        into a sequential list of individual atoms (4CCs)

        input: raw_string: Raw parsed string representation of MP4 structure
        output: A list of atoms in the order they appear in the structure
        """
        # Remove all structural markers, including pipes, brackets, and commas
        clean_str = raw_string.replace('|', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
        
        # Since every valid ISOBMFF box identifier is strictly 4 characters long,
        # we can cleanly slice the string into chunks of 4.
        atoms = [clean_str[i:i+4] for i in range(0, len(clean_str), 4)]
        return atoms

    def _generate_ngrams(self, sequence, n):
        """
        Creates a sorted list of unique n-grams (tuples of atoms) from the sequence

        input: sequence: A list of atoms in the order they appear in the structure
               n: The size of the n-gram (e.g., 2 for bigrams)
        output: A sorted list of unique n-grams (as tuples)
        """
        ngrams = []
        
        # Slide a window of size 'n' across the sequence of atoms
        for i in range(len(sequence) - n + 1):

            # Extract 'n' consecutive atoms and store them as an immutable tuple
            ngram = tuple(sequence[i:i+n])
            ngrams.append(ngram)
            
        # Convert to a set to remove duplicates, then back to a sorted list
        # Sorting ensures determinism when debugging or exporting the digest
        unique_sorted_ngrams = sorted(list(set(ngrams)))
        return unique_sorted_ngrams

    def _build_digest(self):
        """
        Constructs the final digest dictionary containing the metadata and 
        the feature sets and uses the global NGRAM_SIZES configuration

        input: None (relies on the instance's atom_sequence and filename)
        output: A dictionary with the filename, total atoms parsed, and the n-gram features
        """
        digest_dict = {
            "filename": self.filename,
            "total_atoms_parsed": len(self.atom_sequence),
            "features": {}
        }

        # Generate and populate the n-grams for each requested size defined in the global config
        for n in NGRAM_SIZES:
            digest_dict["features"][n] = self._generate_ngrams(self.atom_sequence, n)
            
        return digest_dict

    def get_digest(self):
        """
        Returns the generated Similarity Digest as a Python dictionary.
        """
        return self.digest


class SimilarityCalculator:
    """
    Calculates statistical similarity metrics (Jaccard, Tversky) between 
    two Mp4StructureDigest objects based on their extracted n-gram sets.
    """

    def __init__(self, digest_a, digest_b):
        """
        Initializes the calculator with two generated digests, 
        the order of the digests matters for Tversky but not for Jaccard
        
        input: digest_a: The dictionary from the first Mp4StructureDigest (Ground Truth/Evidence)
               digest_b: The dictionary from the second Mp4StructureDigest (Test File)
        """
        self.digest_a = digest_a
        self.digest_b = digest_b

    def calculate_jaccard(self, n):
        """
        Calculates the symmetrical Jaccard coefficient for a specific 
        n-gram size and the order of files does not matter in Jaccard, 
        as it is a pure set-based similarity measure
        
        input: n: The n-gram size to evaluate (e.g., 2 for Bigrams)
        output: Float representing the similarity score (0.0 to 1.0)
        """
        # Extract the features and convert them to Python Sets for highly optimized math operations
        set_a = set(self.digest_a["features"].get(n, []))
        set_b = set(self.digest_b["features"].get(n, []))

        # Edge Case: Both files yielded zero features of this size
        if not set_a and not set_b:
            return 1.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        # Edge Case: Safeguard against ZeroDivisionError
        if union == 0:
            return 0.0

        # Return the Jaccard coefficient, which is the size of the intersection divided by the size of the union of the sets
        return float(intersection) / float(union)

    def calculate_tversky(self, n):
        """
        Calculates the asymmetrical Tversky index for a specific n-gram 
        size using the global TVERSKY_ALPHA and TVERSKY_BETA variables 
        to weight the importance of False Negatives and False Positives 
        (order of files matters)

        input: n: The n-gram size to evaluate (e.g., 2 for Bigrams)
        output: Float representing the similarity score (0.0 to 1.0)
        """
        # Extract the features and convert them to Python Sets
        set_a = set(self.digest_a["features"].get(n, []))
        set_b = set(self.digest_b["features"].get(n, []))

        # Edge Case: Both files yielded zero features of this size
        if not set_a and not set_b:
            return 1.0

        intersection = len(set_a.intersection(set_b))
        diff_a = len(set_a.difference(set_b)) # Elements only in A (False Negatives)
        diff_b = len(set_b.difference(set_a)) # Elements only in B (False Positives)

        # Apply the global weighting parameters
        denominator = intersection + (TVERSKY_ALPHA * diff_a) + (TVERSKY_BETA * diff_b)

        # Edge Case: Safeguard against ZeroDivisionError
        if denominator == 0:
            return 0.0

        # Return the Tversky index, which is the size of the intersection divided by the weighted sum of the intersection and differences
        return float(intersection) / float(denominator)


# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    
    # Example 1: Samsung Galaxy S3 mini (Reference/Ground Truth)
    str_samsung = "ftyp|mdat|moov(mvhd,udta(smrd,smta),trak(tkhd,mdia(mdhd,hdlr,minf(vmhd,dinf(dref),stbl(stsd(avc1(avcC)),stts,stss,stsz,stsc,stco)))),trak(tkhd,mdia(mdhd,hdlr,minf(smhd,dinf(dref),stbl(stsd(mp4a(esds)),stts,stsz,stsc,stco)))))"
    
    # Example 2: Same Video sent via WhatsApp (Test File)
    str_whatsapp = "ftyp|beam|wide|moov(mvhd,trak(tkhd,free,mdia(mdhd,hdlr,minf(vmhd,dinf(dref),stbl(stsd(avc1(avcC,colr)),stts,stss,sdtp,stsc,stsz,stco)))),trak(tkhd,free,mdia(mdhd,hdlr,minf(smhd,dinf(dref),stbl(stsd(mp4a(esds)),stts,stsc,stsz,stco)))))|mdat"

    # 1. Generate the Digests
    digest_a = Mp4StructureDigest("D01_V_flat_move_0001.mp4", str_samsung).get_digest()
    digest_b = Mp4StructureDigest("D01_V_flatWA_move_0001.mp4", str_whatsapp).get_digest()

    # 2. Output File Information and Sample N-Grams
    print("=== EXTRACTED FEATURES ===")
    for digest in [digest_a, digest_b]:
        print(f"\n--- Digest for {digest['filename']} ---")
        print(f"Total Atoms: {digest['total_atoms_parsed']}")
        
        # Loop through n-gram sizes and print only the first 3 pairs
        for n in NGRAM_SIZES:
            print(f"\n  Extracted {n}-Grams (First 3):")
            for item in digest['features'][n][:3]:
                print(f"  {item}")

    # 3. Instantiate the Calculator
    calc = SimilarityCalculator(digest_a, digest_b)

    # 4. Output the Scores
    print("\n\n=== SIMILARITY ANALYSIS ===")
    print(f"File A (Reference): {digest_a['filename']}")
    print(f"File B (Test):      {digest_b['filename']}")
    print(f"Global Tversky Config: Alpha={TVERSKY_ALPHA}, Beta={TVERSKY_BETA}\n")

    for n in NGRAM_SIZES:
        j_score = calc.calculate_jaccard(n)
        t_score = calc.calculate_tversky(n)
        
        print(f"--- {n}-Gram Scores ---")
        print(f"Jaccard: {j_score:.4f}")
        print(f"Tversky: {t_score:.4f}\n")