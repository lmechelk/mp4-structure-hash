"""
MP4 PARSE

Description:
Extracts a flattened syntactical structural string from MP4 files for the 
purpose of Approximate Matching.

Acknowledgments & Credits:
1. Core Parsing Logic: 
   Based on the concepts of 'mp4atoms.py' by Emanuele Ruffaldi 2024.
   The binary parsing loop (mmap, struct unpacking) was adapted and 
   heavily modified for flat string generation.
   https://gist.github.com/eruffaldi

2. Theoretical Framework (MSSH):
   The extraction methodology and the exclusion of the 'mdat' payload 
   implement the theoretical forensic concepts proposed by Klier & 
   Baier 2025.
   https://github.com/SamKlier/mssh

Modifications:
- Implemented recursive string flattening (Topology to String)
- Added comprehensive allowlist for modern forensic features
- Added 64-bit Large Box (size == 1) support
- Added explicit exclusions for non-ISO compliant leaf nodes
- Added error handling for malformed boxes and parsing anomalies
"""


import mmap   # Memory mapping for efficient, zero-copy file reading
import struct # Unpacking binary data (e.g., MP4 headers) into Python types
import sys    # Handling command-line arguments
import os     # Interacting with the file system (e.g., getting file size)


# Valid types, the extended forensic list of container atoms
CONTAINERS = {
    # Base ISO/IEC 14496-12
    b'moov', b'trak', b'mdia', b'minf', b'stbl', b'udta', b'meta', 
    b'edts', b'dinf', b'stsd', b'mvex', b'moof', b'traf', b'mfra',

    # Extended ISO/IEC 14496-14 and -15
    b'mp4a', b'avc1', b'hev1',
    
    # Quick Time Legacy
    b'gmhd', b'clip', b'matt', b'wave'
}

def parse_boxes(mm, start_offset, end_offset):
    """
    Recursively parses boxes of a mp4 file and returns a nested list
    input: memory-mapped file, start and end offsets
    output: list of box types, with nested structures for containers
    """
    structure = []
    offset = start_offset
    
    while offset < end_offset:
        # A standard header needs at least 8 bytes
        if end_offset - offset < 8:
            break
            
        # Extract 4CC identifier
        box_header = mm[offset:offset+8]
        box_size, box_type = struct.unpack('>I4s', box_header)
        header_size = 8
        
        # Box extends to the end of the file (Size = 0)
        if box_size == 0:
            box_size = end_offset - offset
            
        # 64-bit Large Box (Size = 1) critical for huge mdat atoms
        elif box_size == 1:
            if end_offset - offset < 16:
                break
            box_size = struct.unpack('>Q', mm[offset+8:offset+16])[0]
            header_size = 16

        # Sanity check, box size must be at least the header size
        if box_size < header_size:
            break

        # Box type as string (replace invalid bytes with '?')
        box_type_str = box_type.decode('latin1', errors='replace')

        # Failsafe, if the box type contains null bytes, it's likely a parsing error
        if '\x00' in box_type_str:
            structure.append(".")
            break

        # Calculate the start of the inner content (after the header)
        inner_start = offset + header_size

        # Special handling for 'meta' box: skip the 4 bytes of version/flags
        if box_type == b'meta':
            # Peek the next 4 bytes to check if they are version/flags
            peek_bytes = mm[inner_start:inner_start+4]
            # If they are all zeros, we assume it's the version/flags and skip them
            if peek_bytes == b'\x00\x00\x00\x00':
                inner_start += 4

        # Special handling for 'stsd' box: skip the 8 bytes of version/flags and entry count
        elif box_type == b'stsd':
            inner_start += 8

        # Special handling for Video Sample Entries: skip 78 bytes of video metadata
        elif box_type in (b'avc1', b'hev1'):
            inner_start += 78

        # Special handling for Audio Sample Entries: skip 28 bytes of audio metadata
        elif box_type == b'mp4a':
            inner_start += 28

        # Recursive descent if it is a container
        if box_type in CONTAINERS:
            # Parse the contents of the box
            inner_structure = parse_boxes(mm, inner_start, offset + box_size)
            
            # Formatting e.g. "moov(mvhd,trak(tkhd))"
            if inner_structure:
                structure.append(f"{box_type_str}({','.join(inner_structure)})")
            else:
                structure.append(box_type_str)

        # Non-container leaf nodes are added directly (mdat, ftyp, uuid etc)
        else:
            structure.append(box_type_str)
        
        # Move to the next box
        offset += box_size
        
    # Return the list of box types, with nested structures for containers
    return structure

def get_string(file_path):
    """
    Opens the file, maps it, and starts the parser
    input: file path
    output: flattened string representation of the MP4 structure
    """
    file_size = os.path.getsize(file_path)
    
    with open(file_path, "rb") as f:
        # mmap for high-performance read access without RAM overload
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            root_structure = parse_boxes(mm, 0, file_size)
        
        # Separate top-level atoms with a pipe |
        return "|".join(root_structure)

if __name__ == "__main__":
    # Error handling for missing file argument
    if len(sys.argv) < 2:
        print("Error: Please provide an MP4 file.")
        print("Usage: python mp4_parse.py video.mp4")
        sys.exit(1)
    
    # Main execution block with error handling for file parsing
    file_path = sys.argv[1]
    try:
        
        # Filename
        print("\n--- MP4 PARSE ---")
        print(f"File analyzed: {os.path.basename(file_path)}")

        # Signature
        signature = get_string(file_path)
        print("\n--- Signature ---")
        print(signature)
        
        # Statistics
        print("\n--- Statistics ---")
        print(f"String length: {len(signature)} characters")
        clean_string = signature.replace('|', '').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
        print(f"Total boxes parsed: {len(clean_string) // 4}")
        print(f"Contains 'uuid': {'uuid' in signature}")
        print(f"Parsing Errors: {'.' in signature}\n")
        
    except Exception as e:
        print(f"Critical parsing error: {e}")