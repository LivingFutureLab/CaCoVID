import argparse
import json
import os
import pdb
import re
from copy import deepcopy
from pathlib import Path

import numpy as np


# read json files
def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def write_json(path, data):
    with open(path, "w") as fout:
        json.dump(data, fout)
    print("The format file has been saved at:{}".format(path))
    return


def extract_time(content):
    try:
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            content_answer = eval(content_answer)
            start_time = float(content_answer["start_time"])
            end_time = float(content_answer["end_time"])
            return [[start_time, end_time]]
        else:
            print("wrong prediction: {}".format(content))
            return []
    except:
        return []

def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def evaluate(result_file):
    datas = read_json(result_file)
    num = len(datas)

    # miou
    ious = []
    for k in datas.keys():
        vid, caption, gt = k.split(">>>")
        pred = datas[k]
        gt = eval(gt)
        timestamps = extract_time(pred)
        if len(timestamps) != 1:
            print(f"pred={pred},timestamps={timestamps}")
            timestamps = [[gt[1] + 10, gt[1] + 20]]
        # print(f"GT: {gt}, Pred: {timestamps[0]}")

        ious.append(iou(gt, timestamps[0]))

    Result = {0.3: 0, 0.5: 0, 0.7: 0}
    for c_iou in [0.3, 0.5, 0.7]:
        for cur_iou in ious:
            if cur_iou >= c_iou:
                Result[c_iou] = Result[c_iou] + 1

    print("total {} samples".format(num))
    print("IOU 0.3: {0}\nIOU 0.5: {1}\nIOU 0.7: {2}\nmIOU".format(Result[0.3] * 100 / num, Result[0.5] * 100 / num, Result[0.7] * 100 / num), sum(ious) * 100 / num)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--result_file", default="your_result.json")
    args = parser.parse_args()

    evaluate(args.result_file)