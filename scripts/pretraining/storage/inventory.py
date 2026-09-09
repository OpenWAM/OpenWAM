"""Command-line entry point; implementation lives in the installed package."""
from open_wam.artifacts.publication.inventory import main


if __name__ == "__main__":
    raise SystemExit(main())
