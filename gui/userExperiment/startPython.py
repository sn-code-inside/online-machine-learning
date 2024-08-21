import argparse
from spotpython.utils.file import load_and_run_spot_python_experiment


def main(pickle_file):
    spot_tuner, fun_control, design_control, surrogate_control, optimizer_control, p_open = load_and_run_spot_python_experiment(pickle_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a pickle file.')
    parser.add_argument('pickle_file', type=str, help='The path to the pickle file to be processed.')
    args = parser.parse_args()
    main(args.pickle_file)
