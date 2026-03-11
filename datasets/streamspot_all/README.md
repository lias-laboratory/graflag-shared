# StreamSpot ALL Dataset

System provenance graphs dataset for graph-level anomaly detection.

## Source
- Original: [StreamSpot Data Repository](https://github.com/sbustreamspot/sbustreamspot-data)
- Download: http://www3.cs.stonybrook.edu/~emanzoor/streamspot/
- Paper: "StreamSpot: Detecting Anomalies in Information Flows" (DSN 2016)

## Download
The `all.tsv` file (~2.2GB) exceeds GitHub's LFS size limit and must be downloaded separately:
```bash
cd datasets/streamspot_all/
wget http://www3.cs.stonybrook.edu/~emanzoor/streamspot/all.tsv.gz
gunzip all.tsv.gz
```

## Format
Tab-separated file with one edge per line:
```
source-id    source-type    destination-id    destination-type    edge-type    graph-id
```

## Scenarios (600 graphs total)
| Graph IDs | Scenario | Label |
|-----------|----------|-------|
| 0-99 | YouTube browsing | Benign |
| 100-199 | GMail usage | Benign |
| 200-299 | VGame playing | Benign |
| 300-399 | Drive-by-download | **Attack** |
| 400-499 | Download | Benign |
| 500-599 | CNN browsing | Benign |

## Statistics
- ~89.7 million edges
- 600 graphs
- 100 attack graphs (drive-by-download)
- Heterogeneous provenance graphs (processes, files, network connections)

## Dataset Subsets
StreamSpot supports three dataset configurations:
- `all`: All 600 graphs (default)
- `ydc`: YouTube, Download, CNN + attacks (400 graphs)
- `gfc`: GMail, VGame, CNN + attacks (400 graphs)

## Usage
```bash
# Run with default settings
graflag benchmark -m streamspot -d streamspot_all --build

# Run with YDC subset
graflag benchmark -m streamspot -d streamspot_all --build --params DATASET=ydc

# Adjust chunk length (shingle size)
graflag benchmark -m streamspot -d streamspot_all --build --params CHUNK_LENGTH=50
```

## Key Parameters
- `CHUNK_LENGTH`: Shingle size (default: 10, paper uses 50 for best results)
- `NUM_PARALLEL_GRAPHS`: Number of parallel graphs to process (default: 10)
- `DATASET`: Dataset subset: all, ydc, gfc (default: all)
