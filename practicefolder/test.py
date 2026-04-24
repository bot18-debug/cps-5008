import unittest

from untitled-1 import RegressionKNearstNeighbours 

class TestRegressionKNearstNeighbours :
    def test_k(self):
        model = RegressionKNearstNeighbours(5)
        actual = model


    def test_fit(self):
        
        assert_equal(model.data , x)
        assert_equal (model.data , y)
        

    def test_predict(self):
        x_test = ([1])
        model = RegressionKNearstNeighbours(3)
        x= np.array([1,2])
        y = np.array(["a","b","c"]) 
        model.fit(x,y)
        assert_equal(model.predict(x_test))
        expected = np.array([1])
        
        assert_equal(model.predict(x_test),expected)
    
    