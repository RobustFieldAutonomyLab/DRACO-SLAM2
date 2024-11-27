import pickle

pickle_self = "/home/rfal/Documents/DC-DRACO/animate/test_data/usmma_@.pickle"
pickle_draco = "/home/rfal/Documents/DC-DRACO/animate/usmma/@_0_0.pickle"


def read_pickle(pickle_file_path):
    try:
        with open(pickle_file_path, "rb") as file:
            data = pickle.load(file)
        # print("Pickle file loaded successfully:", data)
        # print(f"mse: {data['mse']}")
        print(f"rmse: {data['rmse']}")
        print(f"icp_success_count: {data['icp_success_count']}")

    except FileNotFoundError:
        print("Error: File not found!")
    except pickle.UnpicklingError:
        print("Error: File is not a valid pickle file!")


print("robot_self: ")
for i in range(1,4):
    read_pickle(pickle_self.replace("@",str(i)))

print("robot_draco: ")
for i in range(1,4):
    read_pickle(pickle_draco.replace("@",str(i)))
