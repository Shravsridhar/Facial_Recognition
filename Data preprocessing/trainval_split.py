import os
import shutil
import numpy as np


tr_path = "/home/shravs/deep_learning/face detection/Datasets/train/steve/images"
names = ['Abhishek','ariana','arijit','arr','atif','Bhashmi','bradley','brian','cox','David','dinklage','ed','emilia','gaga','jenna','jenny','john','kit','kristen','lautner','lisa','maisie','matt','matthew','mickey','nikolaj','pooh','rainn','robert','shravs','sonu','sophie','sri','steve']

v_path = "/home/shravs/deep_learning/face detection/Datasets/val/steve/images"
files = os.listdir(tr_path)
for i in files:
  if np.random.rand(1) < 0.2:
     shutil.move(tr_path+'/'+i,v_path+'/'+i)

