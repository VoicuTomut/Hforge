from prettytable import PrettyTable

def main():
    table = PrettyTable(["Type", "Last model save", "New prediction"])
    print(table)


if __name__ == "__main__":
    main()