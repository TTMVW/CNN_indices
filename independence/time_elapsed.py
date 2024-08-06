import time

# Time stamp closure 
#        Note. 
#         Probably should put all timed code into functions and decorate 
#        see here https://www.python-engineer.com/posts/measure-elapsed-time/ 
def make_time_stamp():
    start = time.time()
    def elapsed():
        nonlocal start
        now = time.time()
        elapsed = now - start
        start = now
        return elapsed
    return elapsed
    
if __name__ == "__main__":
    time_elapsed = make_time_stamp()
    _ = time_elapsed() 
    elapsed = time_elapsed()
    print(elapsed)
