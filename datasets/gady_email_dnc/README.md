# GADY Email-DNC Dataset

This directory contains the Email-DNC (Democratic National Committee) email network dataset for use with the GADY method.

## Data Format

The `email-dnc.edges` file is a comma-separated format with the following columns:
- Column 1: Source node ID (sender)
- Column 2: Destination node ID (recipient)
- Column 3: Unix timestamp

Example:
```
419,465,1463507482
869,453,1462337903
943,1151,1463167636
```

## Source

- **Original Source**: http://networkrepository.com/email-dnc.html
- **Dataset**: DNC Email Network
- **Description**: Email communication network from the Democratic National Committee leak

## Download

This dataset was downloaded from the Network Repository:
```bash
curl -L "http://nrvis.com/download/data/dynamic/email-dnc.zip" -o email-dnc.zip
unzip email-dnc.zip
```

## GADY Processing

GADY's `prepare_data.py` will:
1. Read this edges file and parse edges
2. Apply spectral clustering to identify communities
3. Generate anomalous edges as inter-community connections
4. Create train/test splits with injected anomalies

Output files created in method's `data/` directory:
- `email_dnc.csv` - Preprocessed edge list
- `email_dnc{rate}train.npy` - Training data (rate = 0.01, 0.05, 0.1)
- `email_dnc{rate}test.npy` - Test data with injected anomalies
- `ml_email_dnc_node.npy` - Node features (172-dim random)

## Statistics

- Nodes: ~1,891
- Edges: ~39,264
- Timespan: 2016 DNC email period
