"""CLI entry point for the sematic-desktop project."""
from sematic_desktop import list_files


def main() -> None:
    print(list_files("./my_folder"))


if __name__ == "__main__":
    main()
