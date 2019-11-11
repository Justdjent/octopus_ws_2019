# Training pipeline for fast-scnn
## Requirements:
- Python 3.6
## Launch instructions:  

1. `pip install -r requirements.txt`
2. First download dataset using mscoco_download script. (run `python mscoco_download.py -h` for usage instructions)
3. To extract masks and images only for chosen classes run `python dataset_extraction.py -dp /path/to/dataset/dir -sp /path/to/filtered/dataset/dir`
4. To generate csv with train/val split run `python dataset_split.py -dp /path/to/filtered/dataset/dir`
5. To train model run `python train.py /path/to/filtered/dataset/dir`.
