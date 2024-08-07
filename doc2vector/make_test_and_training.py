"""
    Preparing test and train from the Learnign Outcomes.
    Take the file containing all the Learning Outcomes 
    turn it into two files - with one entry per line.
    
    The LO file will be loaded into a bag in random order. 
    but an index file will be made that maps the line to an original line.
    
    The test file will be made from one in 10 of the randomised 
"""
import sys
import re
import random

def load_file():
    bag = {}
    
    whole_file = sys.stdin.read()
    lines = re.split(r'\n',whole_file)
    def get_tuple(line):
        sp = line.split(',"')
        return (sp[0],sp[1][:-1])
        
    learning_outcomes = [ get_tuple(line) for line in lines if '"' in line] #skip headers - they dont have a qouted string them
    bag = [value for value in range(0,len(lines))]
    count = 1
    course_location = []
    random_corpora = []
    while count < len(lines):
         choice = random.randrange(0,len(lines) - count )
         location = bag.pop(choice)
         random_corpora.append(learning_outcomes[location])
         count+=1
        
    
    print(learning_outcomes, len(learning_outcomes))
    print(random_corpora, len(random_corpora))
    
        
    
load_file()