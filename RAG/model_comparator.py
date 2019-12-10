import numpy as np
from scipy import stats

import sys
import argparse

def get_accuracies(file_name,task = 'gender'):
    '''
    extract accuracies and return list or nested list of accuracies
    :param file_name:
    :param task: gender, race or multitask
    :return:
    '''
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [l.decode('ascii') for l in lines]

        if task == 'multitask':
            gender_accuracies = [float(l.decode('ascii').split(" ")[-1][:-1]) for l in lines if
                                 ('Average gender accuracy:' in l.decode('ascii'))]

            race_accuracies = [float(l.decode('ascii').split(" ")[-1][:-1]) for l in lines if
                               ('Average race accuracy:' in l.decode('ascii'))]

            return [gender_accuracies, race_accuracies]

        else:
            accuracies = [float(l.decode('ascii').split(" ")[-1][:-1]) for l in lines if
                          ('Average ' + task + ' accuracy:' in l.decode('ascii'))]

            return accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gender_logs", type=str, default=None, help="Experiments log for gender model")
    parser.add_argument("--race_logs", type=str, default=None, help="Experiments log for gender model")
    parser.add_argument("--multitask_logs", type=str, default=None, help="Experiments log for multitask model")

    args = parser.parse_args()

    gender_acc0 = None
    race_acc0 = None
    gender_acc1 = None
    race_acc1 = None

    if args.gender_logs:
        gender_acc0 = get_accuracies(args.gender_logs, 'gender')
    if args.race_logs:
        race_acc0 = get_accuracies(args.race_logs, 'race')

    if args.multitask_logs:
        gender_acc1, race_acc1 = get_accuracies(args.multitask_logs, 'multitask')

    if gender_acc0 and gender_acc1:
        print ('Two sample, two-tail t test for gender models')
        print (stats.ttest_ind(gender_acc0, gender_acc1))

    if race_acc0 and race_acc1:
        print('Two sample, two-tail t test for race models')
        print(stats.ttest_ind(race_acc0, race_acc1))
