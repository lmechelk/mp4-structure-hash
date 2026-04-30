"""
MP4 PARSE 2.0
Lucie Mechelk, 2026
https://github.com/lmechelk/mp4-structure-hash

Description:
Extracts a syntactical structural tree from MP4 files for the purpose of 
Approximate Matching. Can be run in batch mode to process large datasets.
The output includes a nested dictionary tree representing the box structure, 
a flattened string signature, and statistics about the parsing process.
The formats for output include console, JSON, and CSV, with detailed error 
handling for malformed files.

Acknowledgments & Credits:
1. Core Parsing Logic: 
   Based on the concepts of 'mp4atoms.py' by Emanuele Ruffaldi 2024.
   The binary parsing loop (mmap, struct unpacking) was adapted and 
   heavily modified to build nested structural representations.
   https://gist.github.com/eruffaldi

2. Theoretical Framework (MSSH):
   The extraction methodology and the exclusion of the 'mdat' payload 
   implement the theoretical forensic concepts proposed by Klier & 
   Baier 2025.
   https://github.com/SamKlier/mssh
   https://www.sciencedirect.com/science/article/pii/S2666281725001167
"""

# ==========================================
# IMPORTS
# ==========================================

import mmap     # For memory-mapped file access to handle large files efficiently
import struct   # For binary data handling and unpacking
import sys      # For error handling and console output
import os       # For file handling and path operations
import argparse # For CLI argument parsing
import json     # For JSON output
import csv      # For CSV output


# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# Choose between "json", "csv", "console" for default output
DEFAULT_OUTPUT_FORMAT = "json"

# Valid types, the extended forensic list of mp4 container atoms
CONTAINERS = {

    # Base ISO/IEC 14496-12
    b'moov', b'trak', b'mdia', b'minf', b'stbl', b'udta', b'meta', 
    b'edts', b'dinf', b'stsd', b'mvex', b'moof', b'traf', b'mfra',

    # Extended ISO/IEC 14496-14 and -15
    b'mp4a', b'avc1', b'hev1',

    # Quick Time Legacy
    b'gmhd', b'clip', b'matt', b'wave'
}

# ==========================================
# CLASSES
# ==========================================

class Mp4Parser:
    """
    Parses a single MP4 file and builds a deep nested dictionary tree 
    representing its internal box structure.

    input: file path to an MP4 file
    output: a dictionary containing the filename, parsing status, statistics,
    """
    def __init__(self, file_path):
        
        self.file_path = file_path                      # Store the full path for file access 
        
        # Extract filename, parent directory, and grandparent directory
        name = os.path.basename(file_path)
        parent = os.path.basename(os.path.dirname(file_path))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        if grandparent and parent:
            self.filename = f"{grandparent}/{parent}/{name}"
        else:
            self.filename = file_path
        
        self.tree = []                                  # This will hold the nested structure of boxes as a list of dictionaries
        self.signature_string = ""                      # This will hold the flattened string representation of the structure (nice to have)
        self.statistics = {}                            # Statistics about the parsing process (e.g., total boxes, presence of 'uuid', parsing errors)
        
    def parse(self):
        """
        Executes the parsing process and populates the tree and statistics.
        Returns a dictionary representing the processing result.
        """
        try:
            file_size = os.path.getsize(self.file_path)
            
            with open(self.file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    self.tree = self._parse_boxes(mm, 0, file_size)
                    
            # Generate the string representation from the nested tree (just for nice output, not used for matching)
            self.signature_string = "|".join(self._build_flat_string(self.tree))
            
            # Calculate statistics (e.g., total boxes, presence of 'uuid', parsing errors)
            clean_str = self.signature_string.replace('|', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
            self.statistics = {
                "total_boxes": len(clean_str) // 4,
                "contains_uuid": "uuid" in self.signature_string,
                "parsing_errors": "." in self.signature_string
            }
            
            # Return the structured result
            return {
                "filename": self.filename,
                "status": "success",
                "statistics": self.statistics,
                "signature": self.signature_string,
                "tree": self.tree
            }
            
        # Handle files that are corrupt, not found, or entirely unreadable
        except Exception as e:

            return {
                "filename": self.filename,
                "status": "error",
                "error_message": str(e),
                "statistics": {"total_boxes": 0, "contains_uuid": False, "parsing_errors": True},
                "signature": "",
                "tree": []
            }

    def _parse_boxes(self, mm, start_offset, end_offset):
        """
        Recursively parses valid MP4 boxes (as in CONTAINERS) and returns a list of dictionaries (nodes).
        The standard header is 8 bytes (size + type), but we also handle extended size (16 bytes) and 
        special cases for certain box types (e.g., 'meta', 'stsd', 'avc1', 'hev1', 'mp4a').

        input:
        - mm: memory-mapped file object for efficient access    
        - start_offset: where to start parsing in the file
        - end_offset: where to stop parsing for this level of recursion
        output: a list of dictionaries representing the boxes at this level, with nested 'children' for containers
        """
        nodes = []              # This will hold the nodes at the current level of recursion
        offset = start_offset   # We start parsing from the given offset and continue until we reach the end_offset for this level
        
        # The main parsing loop: we read box headers, determine their size and type, and then either recurse into them 
        # (if they are containers) or just add them as leaf nodes.
        while offset < end_offset:
            if end_offset - offset < 8:
                break
            
            # Read the box header (size and type) with the standard 8-byte header, and handle extended size if needed
            # First 4 bytes: box size (32-bit unsigned int), next 4 bytes: box type (4-character code)
            box_header = mm[offset:offset+8]
            box_size, box_type = struct.unpack('>I4s', box_header)
            header_size = 8
            
            # Handle special cases for box size (0 means until end of file, 1 means extended size in the next 8 bytes)
            if box_size == 0:
                box_size = end_offset - offset
            
            # Handle 64-bit Large Box (Size = 1) critical for huge mdat atoms
            elif box_size == 1:
                if end_offset - offset < 16:
                    break
                box_size = struct.unpack('>Q', mm[offset+8:offset+16])[0]
                header_size = 16

            # Sanity check: box size must be at least the header size
            if box_size < header_size:
                break

            box_type_str = box_type.decode('latin1', errors='replace')

            # Failsafe: Offset desynchronization. if the box type contains null bytes, it's likely a parsing error, 
            # we add a placeholder (".") and break to avoid infinite loops
            if '\x00' in box_type_str:
                nodes.append({"type": "."})
                break

            inner_start = offset + header_size

            # Special handling for certain box types to skip version/flags or other non-structural bytes
            # meta: skip 4 bytes if they are all zeros (version/flags)
            if box_type == b'meta':
                peek_bytes = mm[inner_start:inner_start+4]
                if peek_bytes == b'\x00\x00\x00\x00':
                    inner_start += 4
            
            # stsd: skip 8 bytes (version/flags + entry count)
            elif box_type == b'stsd':
                inner_start += 8
            
            # avc1/hev1: skip 78 bytes of reserved fields and codec-specific data
            elif box_type in (b'avc1', b'hev1'):
                inner_start += 78
            
            # mp4a: skip 28 bytes of reserved fields and codec-specific data
            elif box_type == b'mp4a':
                inner_start += 28

            # Build Node (dictionary) for the current box
            node = {"type": box_type_str}

            # Recursion if container
            if box_type in CONTAINERS:
                children = self._parse_boxes(mm, inner_start, offset + box_size)
                if children:
                    node["children"] = children

            nodes.append(node)
            offset += box_size
        
        return nodes

    def _build_flat_string(self, nodes):
        """
        Recursively flattens the deep dictionary tree back into the familiar 
        topology string format (e.g., "moov(mvhd,trak(tkhd))"), just for display purposes.

        input: a list of nodes (dictionaries) at the current level
        output: a list of strings representing the box types at this level, with nested structures for containers
        """
        parts = []
        for node in nodes:
            if "children" in node:
                children_str = ",".join(self._build_flat_string(node["children"]))
                parts.append(f"{node['type']}({children_str})")
            else:
                parts.append(node["type"])
        return parts


class BatchAnalyzer:
    """
    Handles the I/O operations, iterating through files, and exporting results.

    input: list of file paths, output format, optional output file name
    output: prints results to console or exports to JSON/CSV depending on configuration
    """

    def __init__(self, file_paths, output_format, output_file=None):
        """ 
        Initializes the batch analyzer with the given configuration. 
        """
        self.file_paths = file_paths        # List of file paths to process
        self.output_format = output_format  # Desired output format ("console", "json", "csv")
        self.output_file = output_file      # Optional custom output file name for JSON/CSV exports
        self.results = []                   # This will hold the results for all processed files, which can be printed or exported at the end

    def run(self):
        """
        Main execution method: iterates through the provided file paths, processes each file, and collects results. 
        Finally, it calls the export method to output the results in the desired format.
        """
        for path in self.file_paths:

            # Check if file exists to prevent hard crashes
            if not os.path.isfile(path):
                print(f"Warning: File not found -> {path}", file=sys.stderr)
                self.results.append({
                    "filename": os.path.basename(path) if path else "Unknown",
                    "status": "error",
                    "error_message": "File not found",
                    "statistics": {"total_boxes": 0, "contains_uuid": False, "parsing_errors": True},
                    "signature": "",
                    "tree": []
                })
                continue
            
            # Parse the file
            parser = Mp4Parser(path)
            result = parser.parse()
            self.results.append(result)
            
            if result["status"] == "error":
                print(f"Warning: Parsing failed for {result['filename']} -> {result['error_message']}", file=sys.stderr)

        self._export()

    def _export(self):
        """
        Exports the collected results in the desired format. For console output, it prints detailed statistics for each file. 
        For JSON/CSV, it writes the results to a file.
        """
        if self.output_format == "console":
            self._print_console()
        elif self.output_format == "json":
            self._export_json()
        elif self.output_format == "csv":
            self._export_csv()

    def _print_console(self):
        """
        Prints the results to the console in a human-readable format, including statistics and error messages if applicable.
        """
        total_files = len(self.results)
        for i, res in enumerate(self.results, start=1):
            print(f"\n--- MP4 PARSE [{i}/{total_files}] ---")
            print(f"File analyzed: {res['filename']}")
            print(f"Status: {res['status'].upper()}")
            
            if res['status'] == "success":
                print(res['signature'])
                print(f"String length: {len(res['signature'])} characters")
                print(f"Total boxes parsed: {res['statistics']['total_boxes']}")
                print(f"Contains 'uuid': {res['statistics']['contains_uuid']}")
                print(f"Parsing Errors: {res['statistics']['parsing_errors']}\n")
            else:
                print(f"Error Message: {res.get('error_message', 'Unknown error')}\n")

    def _export_json(self):
        """
        Exports the entire results object to a JSON file, which includes all details and nested structures for each processed file.
        """
        # Dump the entire complex result object (including deep trees)
        out_target = self.output_file if self.output_file else "output.json"
        with open(out_target, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"Exported {len(self.results)} records to {out_target}")

    def _export_csv(self):
        """
        Exports a flat CSV. Includes the deep JSON tree as a minified string column.
        """
        out_target = self.output_file if self.output_file else "output.csv"
        with open(out_target, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';') # Use semicolon for EU excel compatibility
            
            # Header information
            writer.writerow(["Filename", "Status", "Total_Boxes", "Contains_UUID", "Parsing_Errors", "Signature", "Error_Message", "Tree_JSON"])
            
            for res in self.results:
                stats = res.get("statistics", {})
                
                # Jsonify the tree
                tree_json_str = json.dumps(res.get("tree", []), separators=(',', ':')) if res.get("status") == "success" else "[]"
                
                # Write the row
                writer.writerow([
                    res.get("filename", ""),
                    res.get("status", ""),
                    stats.get("total_boxes", 0),
                    stats.get("contains_uuid", False),
                    stats.get("parsing_errors", False),
                    res.get("signature", ""),
                    tree_json_str,
                    res.get("error_message", "")
                ])
        print(f"Exported {len(self.results)} records to {out_target}")


# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="MP4 Structure Parser for Approximate Matching.")
    parser.add_argument("files", nargs="*", help="Direct paths to one or more MP4 files")
    parser.add_argument("--list", type=str, help="Path to a text file containing a list of MP4 file paths (one per line)")
    parser.add_argument("--format", choices=["console", "c", "json", "j", "csv", "x"], default=DEFAULT_OUTPUT_FORMAT, help="Output format (default: json)")
    parser.add_argument("--output", type=str, help="Custom output filename for json/csv exports")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Shorthands
    format_mapping = {"c": "console", "console": "console", 
                      "j": "json", "json": "json", 
                      "csv": "csv", "x": "csv"}
    args.format = format_mapping[args.format]
    
    # Gather all target files
    target_files = []
    
    # Files or paths have been provided
    if args.files:
        for path in args.files:
            if os.path.isfile(path):
                
                # Check if it's an MP4 or MOV
                if path.lower().endswith(('.mp4', '.mov')):
                    target_files.append(path)
                else:
                    print(f"Warning: Ignored unsupported file -> {path}", file=sys.stderr)
            
            elif os.path.isdir(path):
                
                # If it's a directory, scan it
                print(f"Scanning directory: {path} ...")
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.mov')):
                            target_files.append(os.path.join(root, file))
            else:
                print(f"Warning: Path not found -> {path}", file=sys.stderr)
    
    # List file has been provided
    if args.list:
        if os.path.isfile(args.list):
            with open(args.list, 'r', encoding='utf-8') as f:
                target_files.extend([line.strip() for line in f if line.strip()])
        else:
            print(f"Error: The list file '{args.list}' was not found.", file=sys.stderr)
            sys.exit(1)
            
    # No files have been provided
    if not target_files:
        parser.print_help()
        print("\nError: No MP4 files found. Please provide valid files, folders, or a --list.", file=sys.stderr)
        sys.exit(1)

    # Run the batch analysis
    analyzer = BatchAnalyzer(target_files, args.format, args.output)
    analyzer.run()