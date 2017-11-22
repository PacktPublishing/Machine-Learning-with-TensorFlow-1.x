from prettytable import PrettyTable


def print_variables(variables):
    table = PrettyTable(["Variable Name", "Shape"])
    for var in variables:
        table.add_row([var.name, var.get_shape()])
    print(table)
    print("")


def print_layers(layers):
    table = PrettyTable(["Layer Name", "Shape"])
    for var in layers.values():
        table.add_row([var.name, var.get_shape()])
    print(table)
    print("")


def lines_from_file(filename, repeat=False):
    with open(filename) as handle:
        while True:
            try:
                line = next(handle)
                yield line.strip()
            except StopIteration as e:
                if repeat:
                    handle.seek(0)
                else:
                    raise

if __name__ == "__main__":
    data_reader = lines_from_file("/home/ubuntu/datasets/ucf101/sample.txt", repeat=True)

    for i in range(15):
        print(next(data_reader))
