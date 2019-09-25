'''
This code is part of the publication "On the Origins of Memes by Means of Fringe Web Communities" at IMC 2018.
If you use this code please cite the publication.
'''
import matplotlib.image as mpimg
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt
import requests
import shutil
from PIL import Image
plt.switch_backend('agg')
from matplotlib.backends.backend_pdf import PdfPages
import math
import sys
from scipy.io import savemat
import pickle
from optparse import OptionParser


CLUSTERING_THRESHOLD=8.0
CLUSTERING_MIN_SAMPLES=5
# load data to dictionary
parser = OptionParser()
parser.add_option("-p", "--phashes", dest='phashes', default='phashes.txt', help="file with phashes")
parser.add_option("-d", "--distances", dest='distances', default='phashes-diffs.json',help="file with pairwise distances")
parser.add_option("-m", "--matrix", dest='matrix', default='distance_matrix',help="distances in csr_matrix format")
parser.add_option("-i", "--index", dest='index', default='index_images.p',help="dictionary that includes the mapping between images and index in distance matrix")
parser.add_option("-o", "--output", dest='output', default='clustering_output.txt',help="file that includes clustering output")
parser.add_option("-t", "--test", dest='test', default=False,help="If using test images")

(options, args) = parser.parse_args()

distances_file = options.distances
phashes = options.phashes
clustering_output = options.output
distance_matrix_file = options.matrix
index_image_file = options.index
test = options.test

if test:
    CLUSTERING_MIN_SAMPLES=3
data = json.load(open(distances_file, 'r'))

def load_phashes(fil):
    phash_dict = {}
    with open(fil) as f:
        for line in f:
            el = line.replace('\n','').split('\t')
            image = el[0]
            phash = el[1]
            phash_dict[image] = phash
    return phash_dict

phashes_dict = load_phashes(phashes)


def extract_pairs(pair_text, phashes_dict):
    pairs = []
    els = pair_text.split('-')
    if len(els)>2:
        for i in range(len(els)):
            pair = '-'.join(els[0:i])
            # check if we found first pair
            if pair in phashes_dict:
                pairs.append(pair)
                break
        pair2 = '-'.join(els[i:])
        if pair2 in phashes_dict:
            pairs.append(pair2)
    else:
        return els
    return pairs

# find all pairs and all images from the dict
all_pairs = data.keys()
all_images = []
for pair in all_pairs:
    all_images.extend(extract_pairs(pair, phashes_dict))

all_images = set(all_images)

# create a dict that will provide us with the mapping between the image and the
# index on distance matrix
count = 0
image_index = {}
index_image = {}
for image in all_images:
    image_index[image] = count
    index_image[count] = image
    count+=1
    
# create a distance matrix for the images (use sparse matrix instead of dense
# to avoid memory issues)
n = len(all_images)
distance_matrix = lil_matrix((n, n))

for pair in all_pairs:
    image1, image2 = extract_pairs(pair, phashes_dict)
    distance = data[pair]
    if distance == 0:
        distance = 0.00000000000001
    index1 = image_index[image1]
    index2 = image_index[image2]
    distance_matrix[index1, index2] = distance
    distance_matrix[index2, index1] = distance
savemat(distance_matrix_file, {'M':distance_matrix.tocsr()})
pickle.dump(index_image, open(index_image_file, 'wb'))
print("Done with dumping data...")

def print_clusters_to_file(clustering_output, filename):
    num_clusters = len(dict(Counter(clustering_output.labels_)).keys())
    clusters = clustering_output.labels_.tolist()
    output = open(filename, 'w')
    #output_json = {}
    for k in range(-1, num_clusters):
        output_json= {}
        indices = [i for i, x in enumerate(clusters) if x == k]
        #output.write( "Cluster = %d\n" %k)
        images = []
        if k % 100 == 0:
            print("Calculating medoids. Cluster: %d/%d" %(k, num_clusters))

        if len(indices) > 0:
            for j in indices:
                image = index_image[j]
                images.append(image)
            output_json['cluster_no'] = k
            output_json['images'] = images
            if k!=-1:
                medroid, medroid_path = find_cluster_medroid_phash(clustering_output, k, phashes_dict, distance_matrix, index_image)
                output_json['medroid_phash'] = medroid
                output_json['medroid_path'] = medroid_path
                output.write(json.dumps(output_json) + '\n')
    output.close()
    return output_json

clustering = DBSCAN(eps=CLUSTERING_THRESHOLD, metric='precomputed', n_jobs=8, min_samples=CLUSTERING_MIN_SAMPLES).fit(distance_matrix.tocsr())
num_clusters = len(dict(Counter(clustering.labels_)).keys())
print("Number of clusters  = %d " %(num_clusters-1))



def find_cluster_medroid_phash(cl_output, cluster_num, phash_dict, distance_matr, index_im_dict):
    # get indices of images in cluster
    indices = [i for i, x in enumerate(list(cl_output.labels_)) if x == cluster_num]
    distances = []
    image_names = [index_im_dict[i] for i in indices]
    for i in indices:
        sum_distances = 0.0
        count_distances = 0
        for j in indices:
            if i!=j:
                dist = distance_matrix[i, j]
                # if dist == 0.0 then it means that we dont have distance for the pair in the matrix
                # and we set the distance to a higher value 12 in this case
                if dist == 0.0:
                    dist = 12.0
                sum_distances+=math.pow(dist, 2) # mean squared error
                count_distances+=1
        try:
            mse = sum_distances/ float(count_distances)
            distances.append(mse)
        except:
            distances.append(10000)
    # find index of the image with the min average distance
    ind = distances.index(min(distances))
    ind_in_indices = indices[ind]
    # find the image name that corresponds to the index
    image_name = index_im_dict[ind_in_indices]
    # find phash of the image
    phash = phash_dict[image_name]
    return phash, image_name
    
print("Calculating cluster medoids...")
output_json = print_clusters_to_file(clustering, clustering_output)
print("Clustering output written to %s" %(clustering_output))
