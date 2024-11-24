import pickle
#
# pickle_file_path = "/home/rfal/Documents/DC-DRACO/animate/usmma/test_data/usmma_3.pickle"
pickle_file_path = "/home/rfal/Documents/DC-DRACO/animate/usmma/1_0_0.pickle"
try:
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)
    # print("Pickle file loaded successfully:", data)
    print(f"mse: {data['mse']}")
    print(f"mse: {data['rmse']}")
    print(f"icp_success_count: {data['icp_success_count']}")

except FileNotFoundError:
    print("Error: File not found!")
except pickle.UnpicklingError:
    print("Error: File is not a valid pickle file!")