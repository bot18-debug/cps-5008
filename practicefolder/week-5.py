import numpy as np
class RegressonDecisionTree: 

    def __init__(self,max_depth = 3,min_sample_split =1 ):

        self._max_depth: int = max_depth
        self._min_samples_split: int = min_samples_split
        self._tree = None

        pass
    def predict(self,x):
        pass
    def fit(self , x,y):
        pass 
    def get_tree(self ):
          pass 
    def split_data(self, x, y, threshold):
         _=self
         less_or_equal = ([],[])
         more = ([],[])
         for feature , target in zip (x,y):
                

            if feature <= threshold:
                  less_or_equal[0].append(feature)
                  less_or_equal[1].append(target)
            else :
                more[0].append(feature)
                more[1].append(target)
         return less_or_equal,more
        
      
    