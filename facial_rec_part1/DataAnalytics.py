import numpy as np

class DataAnalytics():

    def __init__(self, buckets):
        self.Buckets = buckets
        self.Top = 7 
        self.Bottom = 1 

    def Mean(self, data):
        return np.mean(data)

    def Min(self, data):
        return np.min(data)

    def Max(self, data):
        return np.max(data)

    def Distribution(self, data):
        distribution = [[]]
        stepSize = self.Top - self.Bottom / self.Buckets

        #sort the data
        sortedData = sroted(data)

        #initialize the buckets
        for _ in range(self.Buckets):
            bucket = []
            distribution.append(bucket)

        #map out the data values in N time 
        bucketIndex = 0
        for value in enumerate(sortedData):
            
            if value < stepSize * bucketIndex:
                distribution[bucketIndex].append(value) 
            else:
                bucketIndex += 1
                distribution[bucketIndex].append(value) 

        return distribution

    def RunAll(self, data):
        return self.Mean(data), self.Min(data), self.Max(data)#, self.Distribution(data) 