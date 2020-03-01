
import statistics
trump = [939, 1336, 1651, 886, 2057]
biden = [1003, 1604, 1824, 708, 2256]
bernie = [810, 881, 1340, 1156, 1918]
mitch = [743, 913, 1118, 801, 1721]
obama = [1259, 1017, 870, 924, 1999]
hillary = [997, 805, 591, 726, 1215]
sarah = [1102, 879, 245, 1205, 1024]
aoc = [948, 858, 751, 1127, 1608]
betsy = [1104, 1008, 835, 1739, 2013]
warren = [851, 1455, 2073, 1741, 2204]

sources = ['breitbart', 'fox', 'usa', 'huff', 'nyt']
candidates = [('trump', trump), ('biden', biden), ('bernie', bernie), ('mitch', mitch), ('obama', obama), ('hillary', hillary), ('sarah', sarah), ('aoc', aoc), ('betsy', betsy), ('warren', warren)]

# for candidate in candidates:
#     print(candidate[0],'\n')
#     print('min:', min(candidate[1]))
#     print('max:', max(candidate[1]))
#     print('mean:', statistics.mean(candidate[1]), '\n')

for index, source in enumerate(sources): 
    print(source, '\n')
    
    source_list = []
    for candidate in candidates:
        source_list.append(candidate[1][index])
    
    print('min:', min(source_list))
    print('max:', max(source_list))
    print('mean:', statistics.mean(source_list), '\n')