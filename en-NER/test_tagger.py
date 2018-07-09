import sys
sys.path.append("./tagger_top_lookup/")
from evaluate import get_ne

if __name__ == '__main__':
    while True:
        rawinput = raw_input("raw_input >>> ")
        print (get_ne(rawinput))
