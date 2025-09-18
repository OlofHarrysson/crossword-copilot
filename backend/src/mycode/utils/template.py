# Note: Almost all of my python scripts should load .env file, parse arguments, and run the main function. Use this code as a template.

import dotenv

# Note: Import the module namespace to make all functions available and the code more readable.
from mycode.utils import argument_parsing

# Note: Use the logger instead of print statements
from mycode.utils.logger import logger


def main() -> None:
    logger.info("Hello World!")


if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    args = argument_parsing.parse_args(main)
    main(**args)
