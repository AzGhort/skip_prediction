from model import Model
import numpy as np


class LastSkipModel(Model):
    def train(self, set):
        pass

    # In the second half of session, repeats last skip behavior of the first half
    def __call__(self, sf_first, sf_second, tf_first, tf_second):
        # skip 2 of the last song in the first half
        return np.repeat(sf_first[-1][3], len(sf_second))


if __name__ == "__main__":
    import argparse
    from predictor import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", default="..\\training_set_mini", type=str, help="Name of the train log folder.")
    parser.add_argument("--test_folder", default="..\\training_set", type=str, help="Name of the test log folder.")
    parser.add_argument("--tf_folder", default="..\\tf", type=str, help="Name of track features folder")
    args = parser.parse_args()

    model = LastSkipModel()
    predictor = Predictor(model)
    maa = predictor.evaluate_on_files(args.test_folder, args.tf_folder)
    print("Last user skip model achieved " + str(maa) + " mean average accuracy")
