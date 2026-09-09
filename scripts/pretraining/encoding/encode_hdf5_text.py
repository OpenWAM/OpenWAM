"""Command-line entry point; implementation lives in the installed package."""
from open_wam.data.preparation.encoding.encode_hdf5_text import main


if __name__ == "__main__":
    raise SystemExit(main())
