from src.read_data import read_data

def main():
    flow_matrix, distance_matrix = read_data()
    print(flow_matrix, distance_matrix)


if __name__ == "__main__":
    main()
