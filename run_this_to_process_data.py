from functions.data_processor import *
from functions.helpers import load_csv_data_modified

if __name__ == "__main__":
    # Will process the x_train dataset.
    x_train, x_test, y_train, train_ids, test_ids, column_names = load_csv_data_modified("./dataset/")
    new_x_train, features, dict_map = process_data(x_train, column_names)
    save_processed_data(new_x_train, features, dict_map)