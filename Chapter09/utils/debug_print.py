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
