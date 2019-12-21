import os

import csv


def check_att(target_att, att, diff_num):
    diff_count = 0
    for att_name in target_att:
        if target_att[att_name] == att[att_name]:
            continue
        diff_count += 1
        if diff_count > diff_num:
            return False

    return True


train_size = 100000


target = 0
csv_path = os.path.dirname(
    os.path.abspath(__file__)
) + "\\attributes\\img_" + str(target) + ".csv"

target_att = {}
with open(csv_path) as f:
    for i, row in enumerate(csv.reader(f)):
        name_in_csv = row[0].strip()
        type_in_csv = int(row[1].strip())

        target_att[name_in_csv] = type_in_csv


for i in range(0, train_size):
    csv_path = os.path.dirname(
        os.path.abspath(__file__)
    ) + "\\attributes\\img_" + str(i) + ".csv"

    att = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            name_in_csv = row[0].strip()
            type_in_csv = int(row[1].strip())

            att[name_in_csv] = type_in_csv

        if check_att(target_att, att, 7):
            print("No." + str(i))

    if i % 1000 == 0:
        print(str(i) + " imgs")
