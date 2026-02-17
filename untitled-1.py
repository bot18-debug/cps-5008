from imports import np 

class RegressionKNearstNeighbours:


    def __init__(self,k = 0 )->None:

        self._k=k
        self._data :array = None
        self._labels: array = None 
    
    @property 
    def k(self)-> int:
        return self._k
    
    @property 
    def data(self)-> array :
        return self._data
    @property 
    def labels (self)-> array :
        return self._labels


    @k.setter 
    def k(self,k_value:int)->None :
        pass 


    
    def fit(self,x:array,y:array):
        self._data  = x
        self._lables  = y 



    
    def predict(self,x:array ):
        distance = []
        for entry in self._data :
             distance.append(self._distance(x,entry ))

             indexes = np.argsort(distance)[:self._k]
             neighbours = self._

    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
                return np.sqrt(sum(square(p1-p2))) 

           



        


