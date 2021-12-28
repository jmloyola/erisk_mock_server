# Name of the database where all the information is going to be stored.
database_name = "erisk"
# List of challenges we want the server to serve.
challenges_list = ["gambling", "depression"]
# Path to the directory containing the datasets.
dataset_root_path = "datasets"
# Suffix for datasets file names. We assume that the final path to a particular
# corpus file is `dataset_root_path + challenge_name + dataset_file_suffix`.
dataset_file_suffix = "-test-clean.txt"
# Token used to separate posts.
end_of_post_token = "$END_OF_POST$"
# Delays used to report the ranking measures.
chosen_delays_for_ranking = [1, 2, 3]
