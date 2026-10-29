from functions.data_processor import *
from functions.helpers import load_csv_data_modified
import json

if __name__ == "__main__":
    # Will process the x_test dataset requires that you have done x_train already.
    x_train, x_test, y_train, train_ids, test_ids, column_names = load_csv_data_modified("./dataset/")

    with open("./processed_dataset/dict_map.json", "r", encoding="utf-8") as f:
        dict_map = json.loads(f.read())

    new_x_test, features, dict_map = process_test_data(x_test, column_names, dict_map=dict_map)
    save_processed_test_data(new_x_test, features)