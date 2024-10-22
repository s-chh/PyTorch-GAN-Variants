def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()
