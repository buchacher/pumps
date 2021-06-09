import sys

from task1_train import task1_train
from task1_test import task1_test
from task1_predict import task1_predict
from task2_train import task2_train
from task2_test import task2_test
from task2_predict import task2_predict
from task3_train import task3_train
from task3_test import task3_test
from task3_predict import task3_predict
from task4_train import task4_train
from task4_test import task4_test
from task4_predict import task4_predict


def print_usage():
    """Prints usage in case of invalid command-line arguments."""
    print("Usage: python3 main.py <task_id> <train|test|predict>")


if len(sys.argv) != 3:
    print_usage()
    sys.exit(0)

task = sys.argv[1]
procedure = sys.argv[2]

if task == "task1":
    if procedure == "train":
        task1_train()
    elif procedure == "test":
        task1_test()
    elif procedure == "predict":
        task1_predict()
    else:
        print_usage()
elif task == "task2":
    if procedure == "train":
        task2_train()
    elif procedure == "test":
        task2_test()
    elif procedure == "predict":
        task2_predict()
    else:
        print_usage()
elif task == "task3":
    if procedure == "train":
        task3_train()
    elif procedure == "test":
        task3_test()
    elif procedure == "predict":
        task3_predict()
    else:
        print_usage()
elif task == "task4":
    if procedure == "train":
        task4_train()
    elif procedure == "test":
        task4_test()
    elif procedure == "predict":
        task4_predict()
    else:
        print_usage()
else:
    print_usage()
