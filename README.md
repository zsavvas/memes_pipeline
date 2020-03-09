# Memes Processing Pipeline

This repository contain our Python scripts for our meme processing pipeline (see https://arxiv.org/abs/1805.12512 for detailed description on the pipeline). In a nutshell, this code performs clustering of images using pHashes and the DBSCAN algorithm and performs annotation of the clusters using images obtained from a memes encyclopedia (Know Your Meme).


## Prerequisites

The pipeline relies on the following packages:
```
Imagehash library == 3.5 (https://github.com/JohannesBuchner/imagehash)
Tensorflow >= 1.5.0 (https://github.com/tensorflow/tensorflow)
Sklearn
Numpy 
```
For the Imagehash we ***strongly recommend*** using the library's 3.5 version. This is because they changed the pHash generation in versions greater than 3.5, hence our generated phashes will be incompatible with pHashes generated with the versions > 3.5.

## Running the pipeline

Now we will describe how you can run our scripts with a set of test images that are included in the repository. During this test, we will provide a brief description of what each script does.

### Calculating pHashes
```
python calculate_phashes.py --directory test_images
```
This script iterates over all images in the `test_images` directory and calculates their pHashes using the Imagehash library. At the end, you should see a new file called `phashes.txt`, which contains the images path and the generated pHash. Note that you can change the output file using the option `--output`.

### Calculating all pairwise Hamming distances between the pHashes
```
python pairwise_comparisons.py 
```
This script takes as an input the pHashes file (by default uses `phashes.txt`, you can change using the option `--input`) and calculates all the pairwise Hamming distances between the pHashes that are below a specific threshold (by default is 10, see `DISTANCE_THRESHOLD` at line 25). The output of the script is a json file where keys are the pairs of images in the form of `image1-image2` and the value is their Hamming distance based on their pHashes. By default, the output file is `phashes-diffs.json` but you can change it using the option `--output`.

### Clustering using all the pairwise distances
```
python clustering_phashes.py --test True
```
This script takes as an input the phashes file (default is `phashes.txt`, you can change using the option `--phashes`) and the json file with all the pairwise distances (default is `phashes-diffs.json`, you can change using the option `--distances`) and performs a clustering of the images using the DBSCAN algorithm. It also calculates the medoid for each cluster (except cluster -1, which is considered noise) and includes it in the output.
It outputs the clustering output in `clustering_output.txt` (you can change using the option --output), the distances in a matrix format (default is `distance_matrix.mat`, you can change using the option `--matrix`), and a dictionary that includes the mapping between the image and the matrix index (default is `index_images.p`, you can change with option `--index`).
Note that the option `--test True` is used for small datasets and decreases the DBSCAN's `min_samples` argument from 5 (default) to 3.

### Visualizing clusters
```
python visualize_clusters.py
```
This script takes as an input all the output of the `clustering_phashes.py` file and visualizes each cluster in a separate pdf (by default in `clusters_visualization` folder, you can change it using the option `--output`).
Note that in the generation of the pdfs we take into account also the pHashes and if two or more images have the same pHash then we only plot one of the images in the pdf to save space. On top of the pdf we report the overall number of images that are in the cluster irrespectively of their pHashes.

### Annotation via Know Your Meme (KYM)
```
python annotate_via_kym.py
```
This script annotates each of the generated clusters using data collected from the Know Your Meme site (KYM, https://knowyourmeme.com/). Specifically, it relies on the `kym_phashes_classes.txt`, which contains all the pHashes from the images obtained from KYM.
The script takes as an input the phashes file (default is `phashes.txt`, you can change using the option `--phashes`) and the clustering output (default is `clustering_output.txt`, you can change using the option `--clustering`), and it outputs a json file in the format of `"cluster_no-kym_image"`: `"hamming distance"`. 
This will includes all the KYM images that have a hamming distance of less or equal of 8 with a cluster medoid. To change the default distance using the option `--distance`.

## Dataset
The dataset used for the paper below is available at https://zenodo.org/record/3699670#.XmZgkZNKi3A. The dataset contains all the image URLs and pHashes for all the images that were posted on Reddit, Twitter (1% Streaming API), 4chan's Politically Incorrect (/pol/) board, and Gab.

## Reference
If you use or find this source code or dataset useful please cite the following work:

    @inproceedings{zannettou2018origins,
    author = {Zannettou, Savvas and Caulfield, Tristan and Blackburn, Jeremy and De Cristofaro, Emiliano and Sirivianos, Michael and Stringhini, Gianluca and Suarez-Tangil, Guillermo},
    title = {{On the Origins of Memes by Means of Fringe Web Communities}},
    booktitle = {IMC},
    year = {2018}
    }


## Acknowledgments

* This project has received funding from the European Union’s Horizon 2020 Research and Innovation program under the Marie Skłodowska-Curie ENCASE project (Grant Agreement No. 691025).
* We also gratefully acknowledge the support of the NVIDIA Corporation for the donation of the Titan Xp GPUs used for our experiments.
