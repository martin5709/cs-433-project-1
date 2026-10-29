# Welcome to the project!

To setup the environment, install the packages from the `requirements.txt` file. I would recommend creating a virtual environment first.

# Recommended Notebook
* From these, the one with the best results is "try7" which is equivalent to attempt 5.

* try3 --> Attempt 1
* ...
* try 8 --> Attempt 6

# How to run:
* Make sure you are using Python 3.14.4
* Make sure that you create the folders: "dataset", "processed_dataset", "models", where "dataset" needs to contain the dataset itself.
* Create a .venv through the requirements.txt
* Run "python run_this_to_process_data.py"
* Run "python run_this_to_process_test_data.py"
* Run the model of your choice (see "trainer_tryX.ipynb")
* Once done, the relevant model should be created under "models"
* Now, run "generate_submission.ipynb" to generate the submission (remember to specify the correct model_name and correct sub_name to match the model of your choice). For try8 specifically, use the try8.ipynb, it requires that try7 has been done already.