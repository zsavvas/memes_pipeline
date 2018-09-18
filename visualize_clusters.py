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
import pickle 
import sys
from scipy.io import loadmat
plt.switch_backend('agg')
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
import os

parser = OptionParser()
parser.add_option("-c", "--clustering", dest='clustering', default='clustering_output.txt',help="file that includes clustering output")
parser.add_option("-m", "--matrix", dest='matrix', default='distance_matrix.mat',help="distances in csr_matrix format")
parser.add_option("-i", "--index", dest='index', default='index_images.p',help="dictionary that includes the mapping between images and index in distance matrix")
parser.add_option("-o", "--output", dest='output', default='clusters_visualization/',help="directory to put pdfs")

(options, args) = parser.parse_args()

inp_file = options.clustering
distance_matrix_file = options.matrix
index_image_file = options.index
base_dir_output = options.output

distance_matrix =loadmat(distance_matrix_file)['M'].tocsr()
index_image = pickle.load(open(index_image_file, 'rb'))

if not os.path.exists(base_dir_output):
    os.makedirs(base_dir_output)

image_index = {}
for k,v in index_image.items():
    image_index[v] = k

with open(inp_file, 'r') as f:
    for line in f:
        output_json = json.loads(line)
        cluster = output_json['cluster_no']
        if cluster == -1:
            continue
        part = 0   
        pdf = PdfPages(base_dir_output + 'cluster'+str(cluster)+'.pdf')

        images_in_cluster = output_json['images']
        
        images_num = len(images_in_cluster)
        print("Cluster = %d Images = %d" %(cluster, images_num))
        
        plt.figure(figsize=(40 ,30))
        plt.rc('text', usetex=False)
        plt.suptitle( "Images in cluster #" + str(cluster) + " = " + str(len(images_in_cluster)), fontsize=50)

        columns = 4
        print("Fetching images from disk....")
        count=0
        count_in_page=0
        images_added = []
        for i, image in enumerate(images_in_cluster):
            path = image
            img=mpimg.imread(path)
            try:
               flag = False
               for im in images_added:
                   if distance_matrix[image_index[image], image_index[im]] == 0.00000000000001:
                        flag=True
                        break
               if flag==False: 
                    plt.subplot(12 / columns + 1, columns, count % 12 + 1)
                    plt.imshow(img)
                    plt.title(path)
                    images_added.append(image)
                    count+=1
                    count_in_page+=1
               if count % 12 == 0 and flag==False:
                    pdf.savefig()
                    plt.close()
                    plt.figure(figsize=(40,30))
                    count_in_page=0
            except Exception as e:
                print(str(e))
                pass
            if count % 1000 == 0 and count > 0 and flag==False:
                pdf.close()
                break
        try: 
            if count_in_page >0:
                pdf.savefig()
                plt.close()
        except:
            pass
        try:
            pdf.close()
        except:
            pass


