#!/usr/bin/env python
#Converts StackOverflow data dump XML files to CSV
#Modified version of converter from https://github.com/mdamien/stackoverflow-neo4j

import json, sys, os, xmltodict, csv
from os.path import join
from utils import *
import shutil
from tqdm import tqdm
import sys
import re

PATH = sys.argv[1]
TEST=False

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

def replace_keys(row):
	new = {}
	for key,val in row.items():
		new[key.lower().replace('@','')] = val
	return new

def clean(x):
    #neo4j-import doesn't support: multiline (coming soon), quotes next to each other and escape quotes with '\""'
    #return x.replace('\n','').replace('\r','').replace('\\','').replace('"','')
    return re.sub('<[^<]+>', "", x.replace('\n','').replace('\r','').replace('\\','').replace('"',''))# remove tags with re.sub('<[^<]+>', "", x)
    

def open_csv(name):
    return csv.writer(open('csvs/{}.csv'.format(name), 'w',encoding='utf-8'), doublequote=False, escapechar='\\')

try:
    shutil.rmtree('csvs/')
except:
    pass
os.mkdir('csvs')

#Posts
posts = open_csv('posts')
posts_rel = open_csv('posts_rel_raw')
posts_answers = open_csv('posts_answers_raw')
tags_posts_rel = open_csv('tags_posts_rel_raw')
users_posts_rel = open_csv('users_posts_rel_raw')
users_posts_rel.writerow([':START_ID(User)', ':END_ID(Post)'])
posts.writerow(['postId:ID(Post)', 'title', 'body','score:INT','views:INT','comments:INT'])
posts_rel.writerow([':START_ID(Post)', ':END_ID(Post)'])
posts_answers.writerow([':START_ID(Post)', ':END_ID(Post)'])
tags_posts_rel.writerow([':START_ID(Post)', ':END_ID(Tag)'])

file = join(PATH,'Posts.xml')
pbar = tqdm(total=os.path.getsize(file), unit="byte", unit_scale=True)
with open(file, 'r',encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        pbar.update(sys.getsizeof(line))
        try:
            if line.startswith("<row"):
                el = xmltodict.parse(line)['row']
                el = replace_keys(el)
                posts.writerow([
                    el['id'],
                    clean(el.get('title','')),
                    clean(el.get('body','')),
                    clean(el.get('score','')),
                    clean(el.get('viewcount','')),
                    clean(el.get('commentcount','')),
                ])
                if el.get('parentid'):
                    posts_rel.writerow([el['parentid'],el['id']])
                if el.get('acceptedanswerid'):
                    posts_answers.writerow([el['id'],el['acceptedanswerid']])
                if el.get('owneruserid'):
                    users_posts_rel.writerow([el['owneruserid'],el['id']])
                if el.get('tags'):
                    eltags = [x.replace('|','') for x in el.get('tags').split('|')]
                    for tag in [x for x in eltags if x]:
                        tags_posts_rel.writerow([el['id'],tag])
        except Exception as e:
            print('x',e)
        if (TEST and i==10000): break
print(i,'posts ok')


#Users
users = open_csv('users')
users_things = ['displayname', 'reputation:INT', 'aboutme', \
    'websiteurl', 'location', 'profileimageurl', 'views:INT', 'upvotes:INT', 'downvotes:INT']
users.writerow(['userId:ID(User)'] + users_things)
file = join(PATH,'Users.xml')
pbar = tqdm(total=os.path.getsize(file), unit="byte", unit_scale=True)
with open(file, 'r',encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        pbar.update(sys.getsizeof(line))
        try:
            if line.startswith("<row"):
                el = xmltodict.parse(line)['row']
                el = replace_keys(el)
                row = [el['id'],]
                for k in users_things:
                    row.append(clean(el.get(k,'')[:100]))
                users.writerow(row)
        except Exception as e:
            print('x',e)
        if (TEST and i==10000): break
print(i,'users ok')

#Tags
tags = open_csv('tags')
tags.writerow(['tagId:ID(Tag)'])
file = join(PATH,'Tags.xml')
pbar = tqdm(total=os.path.getsize(file), unit="byte", unit_scale=True)
with open(file, 'r',encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        pbar.update(sys.getsizeof(line))
        try:
            if line.startswith("<row"):
                el = xmltodict.parse(line)['row']
                el = replace_keys(el)
                tags.writerow([
                    el['tagname'],
                ])
        except Exception as e:
            print('x',e)
print(i,'tags ok')
