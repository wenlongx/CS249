import csv

"""
Get only the first portion of a csv file
"""

FILENAME = 'test.csv'
CUT_PERCENTAGE = 0.02

line_num = sum([1 for line in open(FILENAME)])

with open(FILENAME, mode='r') as file_original:
    reader = csv.reader(file_original, delimiter=',', quotechar='"')
    cut_file= open("{0}_{1}.csv".format(".".join(FILENAME.split('.')[0:-1]), CUT_PERCENTAGE), mode='w')
    writer = csv.writer(cut_file, delimiter=',', quotechar='"')

    line_count = 0
    for line in reader:
        if line_count < int(line_num * CUT_PERCENTAGE):
            writer.writerow(line)
        else:
            break
        line_count += 1

    cut_file.close()
