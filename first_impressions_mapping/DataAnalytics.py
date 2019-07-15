import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from DataReader import DataReader

class DataAnalytics():

    def __init__(self, buckets):
        self.Buckets = buckets

    def Mean(self, data):
        return np.mean(data)

    def Min(self, data):  
        return min(data)

    def Max(self, data):
        return max(data)

    def Distribution(self, data):
        distribution = [[]]
        bottom = self.Min(data)
        stepSize = (self.Max(data) - bottom) / self.Buckets

        #sort the data
        sortedData = sorted(data)

        #initialize the buckets
        for _ in range(1, self.Buckets):
            bucket = []
            distribution.append(bucket)

        #map out the data values in N time 
        bucketIndex = 0
        for value in sortedData:

            #if value is in the next bucket
            if value > round((stepSize * bucketIndex) + bottom + stepSize, 2):
                bucketIndex += 1

            distribution[bucketIndex].append(value) 

        return distribution, self.Distribution_count(distribution)
    
    def Distribution_count(self, distribution):
        dist_count = {} 

        for index in range(self.Buckets):
            dist_count[index] = len(distribution[index])
        
        return dist_count

    def RunAll(self, data):

        return self.Mean(data), self.Min(data), self.Max(data), self.Distribution(data) 

    def PlotDistribution(self, data):
        
        uniques = np.unique(data)
        plt.hist(data, bins=len(uniques))
        plt.xticks(np.arange(min(data), max(data) + 1, (self.Max(data) - self.Min(data)) / self.Buckets))
        plt.show()

#reader = DataReader()
#(kdef_partition, kdef_labels) = reader.read_kdef()

#flattened_kdef = []
#values = kdef_labels.values()
#for value in values:
#    for item in value:
#        flattened_kdef.append(item)

#dataAnalytics = DataAnalytics(6)
#mean, minVal, maxVal, (distribution, dist_count) = dataAnalytics.RunAll(flattened_kdef)
#dataAnalytics.PlotDistribution(flattened_kdef)
