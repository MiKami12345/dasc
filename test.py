import csv
import os


def test():
    with open('static/csv/driving_questions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        print(csv_reader)

test()