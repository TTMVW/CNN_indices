"""
    Preparing test and train from the Learning Outcomes.
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
    learning_outcomes = []
    whole_file = sys.stdin.read()
    lines = re.split(r'\n',whole_file)
    
    def get_tuple(line):
        sp = line.split(',"')
        return (sp[0],sp[1][:-1])
    
    for line in lines:
        if '"' in line:
            learning_outcomes += [get_tuple(line)]
        else:
            print(line)
            
    # earning_outcomes = [ get_tuple(line) for line in lines if '"' in line] #skip headers - they dont have a quoted string them
    bag = [value for value in range(0,len(lines))]
    count = 1
    random_corpora = []
    while count < len(lines):
          choice = random.randrange(0,len(lines) - count )
          location = bag.pop(choice)
          random_corpora.append(learning_outcomes[location])
          count+=1
        
    
    
    return learning_outcomes, random_corpora
    
        
 
all_lo, random_corpora = load_file()
print("Len all lo",len(all_lo))


# print(random_corpora, len(random_corpora))
print(all_lo)

with open("all_lo.txt", "w") as all_lo_file:
    for line in all_lo:
        all_lo_file.write(f"{line[0]},{line[1]}\n")

with open("training.cor", "w") as training_file, open("test.cor","w") as test_file, open("corpora_tuples.txt","w") as corpora:
        count = 0
        for line in random_corpora:
            if count % 37 == 0 :
                test_file.write(f"{line[1]}\n")
            
            training_file.write(f"{line[1]}\n")
            count += 1
            corpora.write(f"{line}\n")