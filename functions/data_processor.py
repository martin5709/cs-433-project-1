# Purpose: Process the data.

import json
import numpy as np

default_commands = [
    "DROP _STATE _LLCPWT",
    "MAP A _RFHLTH _RACE_G1",
    "DROP HTIN4",
    "MAP B HTM4 WTKG3",
    "DROP _BMI5CAT _RFBMI5",
    "MAP A _CHLDCNT DRNKANY5",
    "MAP C DROCDY3_",
    "MAP A _RFBING5",
    "MAP E _DRNKWEK",
    "MAP A _RFDRHV5",
    "MAP D FTJUDA1_ VEGEDA1_",
    "MAP A _MISFRTN _VEGRESP",
    "MAP D _FRUTSUM _VEGESUM",
    "MAP A _FRTLT1 _TOTINDA",
    "MAP B METVL11_ METVL21_",
    "MAP B MAXVO2_ FC60_",
    "MAP A ACTIN11_ ACTIN21_",
    "MAP D PADUR1_ PADUR2_",
    "MAP E PAFREQ1_ PAFREQ2_",
    "MAP D _MINAC11 _MINAC21",
    "MAP F STRFREQ_",
    "MAP A PAMISS1_",
    "MAP D PAMIN11_ PA1VIGM_",
    "MAP A _PACAT1 _AIDTST3",
    "DONE"
]

# Helping functions
def map_a(column: np.array, column_name: str) -> tuple[list[np.array], list[str]]:
    """Carries out map type A. Will return a tuple containing the modified array and the feature names. Purpose is to create the indicator variables."""
    uniques = np.unique(column)
    output_list = []
    feature_list = []
    if len(uniques) > 50:
        print(f"Warning! Large number of uniques! In total: {len(uniques)}. Column name: {column_name}")
        print(f"Examples: {uniques[:50]}")
    for x in iter(uniques):
        output_list.append(((column == x)*1))
        feature_list.append(f"{column_name}_{x}")
    return output_list, feature_list

def map_b(column: np.array) -> tuple[np.array, float, float]:
    """Unknown, 99999, 99900 to AVG.
    Returns: column, mean, st.d."""
    mean = np.nanmean(column)
    std = np.nanstd(column)
    rules = [99999, 99900]
    for x in rules:
        mask = (column == x)
        column = column * ~mask + mask * mean
    column = np.nan_to_num(column, nan=mean)
    column = (column - mean)/std
    return column, mean, std

def map_c(column: np.array) -> tuple[np.array, float, float]:
    """900 to 0. Missing to 0.
    Returns: column, mean, st.d."""
    mean = np.nanmean(column)
    std = np.nanstd(column)
    rules = [900]
    for x in rules:
        mask = (column == x)
        column = column * ~mask + mask * 0
    column = np.nan_to_num(column, nan=0)
    column = (column - mean)/std
    return column, mean, std

def map_d(column: np.array) -> tuple[np.array, float, float]:
    """Missing to 0.
    Returns: column, mean, st.d."""
    mean = np.nanmean(column)
    std = np.nanstd(column)
    column = np.nan_to_num(column, nan=0)
    column = (column - mean)/std
    return column, mean, std

def map_e(column: np.array) -> tuple[np.array, float, float]:
    """99900 to 0. Missing to 0.
    Returns: column, mean, st.d."""
    mean = np.nanmean(column)
    std = np.nanstd(column)
    rules = [99900]
    for x in rules:
        mask = (column == x)
        column = column * ~mask + mask * 0
    column = np.nan_to_num(column, nan=0)
    column = (column - mean)/std
    return column, mean, std

def map_f(column: np.array) -> tuple[np.array, float, float]:
    """99000 to 0, missing to 0.
    Returns: column, mean, st.d."""
    mean = np.nanmean(column)
    std = np.nanstd(column)
    rules = [99000]
    for x in rules:
        mask = (column == x)
        column = column * ~mask + mask * 0
    column = np.nan_to_num(column, nan=0)
    column = (column - mean)/std
    return column, mean, std

def map_g(column: np.array) -> np.array:
    """NAN Policy. Will just set NANs to 0."""
    column = np.nan_to_num(column, nan=0)
    return column

def check_valid_columns(commands, column_names):
    """Check if the provided columns exist in the column names set."""
    result = True

    for x in commands:
        current_command = x.split()
        if current_command[0] == "DROP":
            result = result and (current_command[1] in column_names)
            result = result and (current_command[len(current_command) - 1] in column_names)
        elif current_command[0] == "MAP":
            result = result and (current_command[2] in column_names)
            result = result and (current_command[len(current_command) - 1] in column_names)
        if not result:
            print(current_command)
            return result

    return result

# Main Functions
def process_data(x: np.array, column_names: list[str], commands: list[str] = default_commands) -> tuple[np.array, list[str], dict[str, list[float]]]:
    """Carry out the specified modifications.
    Assumes that no rows are dropped during transformations.
    Returns:
    1. The values but processed.
    2. The names of these features.
    3. The mappings to be used for mean and standard deviation."""

    assert check_valid_columns(commands, column_names)

    index = 0 # Current command
    active = False # Whether the current command is active
    temp_output = [] # Output to be transformed into the final numpy array
    features = [] # The names of the columns
    dict_map = {} # The columns which were standardised have their mean and std specified here.

    for i in range(x.shape[1]):
        current_command = commands[index].split(" ")
        # Step 1: Activate the command if possible.
        if current_command[0] == "DROP":
            if (not active) and (current_command[1] == column_names[i]):
                active = True
        elif current_command[0] == "MAP":
            if (not active) and (current_command[2] == column_names[i]):
                active = True
        elif current_command[0] == "DONE":
            pass # Do nothing.
        else:
            raise Exception(f"Invalid Command. Received: {current_command[0]}")

        #print(f"Column: {column_names[i]}, Current Rule: {commands[index]} Active: {active}")

        # Step 2: If Active, do mappings
        if active:
            if current_command[0] == "DROP":
                pass # Do nothing.
            elif current_command[0] == "MAP":
                if current_command[1] == "A":
                    output_list, feature_list = map_a(x[:, i], column_names[i])
                    temp_output.extend(output_list)
                    features.extend(feature_list)
                elif current_command[1] == "B":
                    output_col, mean, std = map_b(x[:, i])
                    temp_output.append(output_col)
                    features.append(column_names[i])
                    dict_map[column_names[i]] = [float(mean), float(std)]
                elif current_command[1] == "C":
                    output_col, mean, std = map_c(x[:, i])
                    temp_output.append(output_col)
                    features.append(column_names[i])
                    dict_map[column_names[i]] = [float(mean), float(std)]
                elif current_command[1] == "D":
                    output_col, mean, std = map_d(x[:, i])
                    temp_output.append(output_col)
                    features.append(column_names[i])
                    dict_map[column_names[i]] = [float(mean), float(std)]
                elif current_command[1] == "E":
                    output_col, mean, std = map_e(x[:, i])
                    temp_output.append(output_col)
                    features.append(column_names[i])
                    dict_map[column_names[i]] = [float(mean), float(std)]
                elif current_command[1] == "F":
                    output_col, mean, std = map_f(x[:, i])
                    temp_output.append(output_col)
                    features.append(column_names[i])
                    dict_map[column_names[i]] = [float(mean), float(std)]
                else:
                    raise Exception("Invalid MAP command requested.")
        else:
            temp_output.append(map_g(x[:, i])) # Only map NANs to 0.
            features.append(column_names[i])

        # Step 3: If no longer active, advance to next command, or no check if last command.
        if current_command[0] == "DROP":
            if len(current_command) == 2:
                active = False
                index += 1
            elif len(current_command) == 3:
                if current_command[2] == column_names[i]:
                    active = False
                    index += 1
        elif current_command[0] == "MAP":
            if len(current_command) == 3:
                active = False
                index += 1
            elif len(current_command) == 4:
                if current_command[3] == column_names[i]:
                    active = False
                    index += 1

    return np.column_stack(temp_output), features, dict_map

def save_processed_data(new_x_train, features, dict_map, path: str = "./processed_dataset/"):
    """Take the output of process_data and this will save it."""
    np.savetxt(f"{path}X_processed.csv", new_x_train, delimiter=",")
    with open(f"{path}features.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(features))

    with open(f"{path}dict_map.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict_map))