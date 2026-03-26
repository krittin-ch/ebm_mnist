import argparse
from datetime import datetime


def parse_config():
    parser = argparse.ArgumentParser(description="argparse example")

    parser.add_argument("--name", "-n", type=str, help="your name")
    parser.add_argument("--year_of_birth", "-y", type=int, help="your birth year")
    parser.add_argument(
        "--nectec", action="store_true", help="call this parser if you work at NECTEC"
    )

    args = parser.parse_args()
    return args


def args_print(args):
    print("===== CONFIG =====")
    for k, v in vars(args).items():
        print(f"{k:15}: {v}")
    print("==================\n")


def main():
    args = parse_config()
    args_print(args)

    current_year = datetime.now().year

    print(f"Hi, I am {args.name}.")
    print(f"I am {current_year - args.year_of_birth} years old.")
    print(f"I {'do' if args.nectec else 'do not'} work at NECTEC.")


if __name__ == "__main__":
    """
    Try the following in your terminal/command line
    1. python argparse_example.py -n Krittin -y 2003 --nectec
    2. python argparse_example.py -n Krittin -y 2003

    Also, try `argparse_scripts.sh`

    MacOS/Linux:
    chmod +x argparse_scripts.sh
    ./argparse_scripts.sh or bash argparse_scripts.sh

    Windows (might need git/wsl installation):
    ./argparse_scripts.sh or bash argparse_scripts.sh
    """
    main()
