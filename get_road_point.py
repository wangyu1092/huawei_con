import os
import pandas as pd
import numpy as np
import time
import math


def get_road_point(point_source, point_destination, d_gap=7.0711):
    x1 = point_source[0]
    y1 = point_source[1]
    x2 = point_destination[0]
    y2 = point_destination[1]

    res = []
    res.append((x1, y1))

    if (x2 == x1):
        direction = 1
        if (y2 < y1):
            direction = -1
        res += [(x1, y) for y in range(y1 + d_gap * direction, y2, d_gap * direction)]
        res.append((x2, y2))
        return res
    if (y2 == y1):
        direction = 1
        if (x2 < x1):
            direction = -1
        res += [(x, y1) for x in range(x1 + d_gap * direction, x2, d_gap * direction)]
        res.append((x2, y2))
        return res

    a = (y2 - y1) / (x2 - x1)
    b = y2 - a * x2
    f = lambda x: a * x + b
    fv = lambda y: y / a - b / a
    between = lambda r, s, e: r >= min(s, e) and r <= max(s, e)
    direction = [1, 1]
    if (x2 < x1):
        direction[0] = -1
    if (y2 < y1):
        direction[1] = -1
    x = x1
    y = y1
    gap = d_gap / math.sqrt(1 + a * a)
    while (1):
        # version 1
        # if ( between( f( x + gap*direction[0] ), y, y + gap*direction[1] ) ):
        #     x += gap*direction[0]
        # elif ( between( fv( y + gap*direction[1] ), x, x + gap*direction[0] ) ):
        #     y += gap*direction[1]
        # else:
        #     print("不应该呀")
        #     print(res)
        #     exit()

        # version 2
        # if (abs(a) <= 1):
        #     x += direction[0] * d_gap
        #     y = f(x)
        #     if (between( x , x1,  x2) ):
        #         res.append((x//5*5,y//5*5))
        #     else:
        #         break
        # else:
        #     y += direction[1] * d_gap
        #     x = fv(y)
        #     if (between( y , y1,  y2) ):
        #         res.append((x//5*5,y//5*5))
        #     else:
        #         break

        x += direction[0] * gap
        y = f(x)
        if (between(x, x1, x2)):
            res.append((x // 5 * 5, y // 5 * 5))
        else:
            break

        # if( not between(x, x1, x2) and between(y, y1, y2) ):
        if (x == x2 and y == y2 or not between(x, x1, x2) and not between(y, y1, y2)):
            break
        # res.append((x,y))
        # print([x,y], "\n")
    res.append((x2, y2))
    return res


# def get_kv_point2prop(dataload = "C:/Users/LYS/Desktop/huawei/", file_name = "train.csv"):
def get_kv_point2prop(X, Y, A, B, C):
    # train_data = pd.read_csv(os.path.join(dataload, file_name), usecols=[12,13,14,15,16])
    res = {}
    for i in range(len(X)):
        res[(X[i], Y[i])] = [A[i], B[i], C[i]]
    return res


if __name__ == "__main__":
    print(get_road_point([0, 0], [0, 15]))

    print(get_road_point([0, 0], [15, 0]))

    print(get_road_point([0, 0], [10, -15]))

    print(get_road_point([0, 0], [15, 10]))

    print(get_road_point([0, 0], [-15, -10]))

    print(get_road_point([0, 0], [15, 11]))  # dont do this

    print(get_road_point([0, 0], [-465, 460]))

    print(get_road_point([424515.0, 3376325.0], [424050.0, 3376785.0]))

    print(get_road_point([424515.0, 3376325.0], [424050.0, 3376785.0]))

    tt = time.time()
    # hashtable = get_kv_point2prop()
    print(time.time() - tt)
    print(type(424515.0))
