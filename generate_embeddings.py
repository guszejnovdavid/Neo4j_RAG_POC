#!/usr/bin/env python
#Creates embeddings for StackOverflow posts from previously generated csv file

import csv
from tqdm import tqdm
import sys
from os.path import join, getsize
from langchain_huggingface import HuggingFaceEmbeddings
import concurrent.futures
import numpy as np

PATH = sys.argv[1]
post_id_ind=0
body_ind=2 
batch_size=128
read_batch_size=32*batch_size

# Set up embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cuda'}, encode_kwargs={'batch_size': batch_size})        
    
def round_to_significant_digits(x, p, minval=1e-8): #from Stackoverflow, modified
    x = np.asarray(x)
    if p is None:
        return x
    else:
        x_positive = np.where(np.isfinite(x) & (np.abs(x) > minval), np.abs(x), 10**(p-1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

def convert_vector_to_neo4j_format(vect,n_digits=None):
        return "|".join(vect.astype(str))

def main():
    posts_file_path = join(PATH,'posts.csv')
    target_file_path = join(PATH,'posts_with_embeddings.csv')
    
    pbar = tqdm(total=getsize(posts_file_path), unit="byte", unit_scale=True)
    with open(posts_file_path, 'r',encoding='utf-8') as posts_file:
        reader = csv.reader(posts_file, doublequote=False, escapechar='\\')
        with open(target_file_path, 'w',encoding='utf-8') as target_file:
            writer = csv.writer(target_file,  doublequote=False, escapechar='\\')
            writer.writerow(next(reader)+["embedding:float[]"]) #header
            batch=[]; read_rows=[]; processed_size=0
            for row in reader:
                if (len(row) and len(row[0])):
                    read_rows.append(row)
                    batch.append(row[body_ind])
                    processed_size=processed_size+sys.getsizeof(','.join(row)) #due to how getsizeof is implemented we can't just use getsizeof(row)
                    if (len(read_rows)>=read_batch_size):
                        embeddings = embedding_model.client.encode(batch, **embedding_model.encode_kwargs)
                        #Keep only significant digits (much smaller file size)
                        embeddings = round_to_significant_digits(embeddings, 3)
                        #Save results
                        writer.writerows([old_row + [convert_vector_to_neo4j_format(embedding)] for old_row, embedding in zip(read_rows,embeddings)]) #save to new csv
                        pbar.update(processed_size) #update status
                        batch=[]; read_rows=[];processed_size=0
            #Process final posts
            if (len(batch)):                       
                embeddings = (embedding_model.client.encode(batch, **embedding_model.encode_kwargs)).tolist()
                embeddings = round_to_significant_digits(embeddings, 3)
                writer.writerows([old_row + [convert_vector_to_neo4j_format(embedding)] for old_row, embedding in zip(read_rows,embeddings)]) #save to new csv
                        
                        
 
import cProfile
if __name__ == "__main__":
    main()
    #cProfile.run('main()','profile_stats')

