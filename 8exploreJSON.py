# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:58:01 2020

@author: andre
EXPLORE JSON FILE
This .json file I am using contains one JSON object in each line as per the specification.
"""
import json
import ijson
import pandas as pd
from pprint import pprint

# You should pass the file contents (i.e. a string) to json.loads(), not the file object itself
with open('avro-issues.json','r') as json_file:
    for line in json_file:    
        alert_dict = json.loads(line)
        objects = ijson.items(alert_dict,'transitions')
        transitions = list(objects)

        
# with pandas
with open('avro-issues.json','r') as json_file:
    alert_df = pd.read_json(json_file, lines=True)    

      
# these reads the json as a long string (do not print)
with open('avro-issues.json', 'r', encoding='utf-8') as data_file:    
    data = data_file.read()

    
##################################################
# each line of the JSON files contains a dictionary
# JSON files has 1458 lines = N# alerts in our dataset
    
list_of_alerts = []
for line in open('avro-issues.json', 'r'):
    list_of_alerts.append(json.loads(line))

len(list_of_alerts)
#for dictionary in list_of_alerts:
    
    
    
##################################################
# general approach
objects = []
for obj in open('avro-issues.json', 'r'):
    objects.append(json.loads(obj))

for obj in objects[0]:
    pp_json(obj)
    
######################################################
# Use this function and don't sweat having to remember if your JSON is a str or dict again - 
# just look at the pretty print:

def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        pprint(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
    else:
        pprint(json.dumps(json_thing, sort_keys=sort, indent=indents))
    return None
######################################################
# print values of single keys in dict: objects[...]

pp_json(objects[0]['transitions'])
# all transition fields are empty!!

for i in range(len(objects)):
    objects[i]

keys = objects[0].keys()
