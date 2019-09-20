import os
import pandas as pd
import numpy as np
import time

def get_road_point(point_source, point_destination, gap = 5):
    x1 = point_source[0]
    y1 = point_source[1]
    x2 = point_destination[0]
    y2 = point_destination[1]
    if (x2 == x1):
        direction = 1
        if (y2 < y1):
            direction = -1
        return [ (x1, y) for y in range(y1 + gap*direction, y2, gap*direction)]
    if (y2 == y1):
        direction = 1
        if (x2 < x1):
            direction = -1
        return [ (x, y1) for x in range(x1 + gap*direction, x2, gap*direction)]

    a = (y2 - y1) / (x2 - x1)
    b = y2 - a * x2
    f = lambda x: a*x+b
    fv = lambda y: y/a-b/a
    between = lambda r,s,e: r >= min(s, e) and r <= max(s, e)
    res = []
    direction = [1,1]
    if (x2 < x1):
        direction[0] = -1
    if (y2 < y1):
        direction[1] = -1
    x = x1
    y = y1
    #res.append((x,y))
    while (1):
        if ( between( f( x + gap*direction[0] ), y, y + gap*direction[1] ) ):
            x += gap*direction[0]
        elif ( between( fv( y + gap*direction[1] ), x, x + gap*direction[0] ) ):
            y += gap*direction[1]
        else:
            print("不应该呀")
            print(res)
            exit()

        #if( not between(x, x1, x2) and between(y, y1, y2) ):
        if (x == x2 and y == y2 or not between(x, x1, x2) and not between(y, y1, y2)):
            break
        res.append((x,y))
        #print([x,y], "\n")

    return res

def get_kv_point2prop(dataload = "C:/Users/LYS/Desktop/huawei/", file_name = "train.csv"):
    train_data = pd.read_csv(os.path.join(dataload, file_name), usecols=[12,13,14,15,16])
    res = {}
    for index, row in train_data.iterrows():
        res[(row[0], row[1])] =  [row[2], row[3], row[4]]
    return res

if __name__ == "__main__":
    print(get_road_point([0,0], [0,15]))

    print(get_road_point([0,0], [15,0]))

    print(get_road_point([0,0], [10,-15]))
    
    print(get_road_point([0,0], [15,10]))

    
    print(get_road_point([0,0], [-15,-10]))

    
    print(get_road_point([0,0], [15,11])) #dont do this

    
    print(get_road_point([0,0], [-465,460]))

    print(get_road_point([424515.0, 3376325.0], [424050.0, 3376785.0]))

    tt = time.time()
    #hashtable = get_kv_point2prop()
    print(time.time() - tt)
    print(type(424515.0))
