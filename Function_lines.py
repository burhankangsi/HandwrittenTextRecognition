import cv2
import numpy as np


def findLines(bw_image, LinesThres):
    # making horizontal projections
    horProj = cv2.reduce(bw_image, 1, cv2.REDUCE_AVG)

    # make hist - same dimension as horProj - if 0 (space), then True, else False
    th = 0;  # black pixels threshold value. this represents the space lines
    hist = horProj <= th;

    # Get mean coordinate of white white pixels groups
    ycoords = []
    y = 0
    count = 0
    isSpace = False

    for i in range(0, bw_image.shape[0]):
        if not isSpace:
            if hist[i]:
                isSpace = True
                count = 1
                y = i

        else:
            if not hist[i]:
                isSpace = False
                if count >= LinesThres:
                    ycoords.append(y / count)
            else:
                y = y + i
                count = count + 1

    ycoords.append(y / count)
    # returns y-coordinates of the lines found
    return ycoords


def LinesMedian(bw_image):
    # making horizontal projections
    horProj = cv2.reduce(bw_image, 1, cv2.REDUCE_AVG)

    # make hist - same dimension as horProj - if 0 (space), then True, else False
    th = 0;  # black pixels threshold value. this represents the space lines
    hist = horProj <= th;

    # Get mean coordinate of white white pixels groups
    ycoords = []
    y = 0
    count = 0
    isSpace = False
    median_count = []
    for i in range(0, bw_image.shape[0]):
        if not isSpace:
            if hist[i]:
                isSpace = True
                count = 1
                # y = 1

        else:
            if not hist[i]:
                isSpace = False
                median_count.append(count)
            else:
                # y = y + i
                count = count + 1

    median_count.append(count)
    # ycoords.append(y / count)
    # returns counts of each blank rows of each of the lines found
    return median_count


def get_lines_threshold(percent, img_for_det):
    ThresPercent = percent
    LinMed = LinesMedian(img_for_det)
    LinMed = sorted(LinMed)
    LinesThres = LinMed[len(LinMed) // 3] * (ThresPercent // 100.0)
    LinesThres = int(LinesThres)
    return LinesThres
