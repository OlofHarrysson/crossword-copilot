import uvicorn
import dotenv

from mycode.utils import argument_parsing


def start_server():
    uvicorn.run(
        "mycode.api.app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=True,
        workers=0,
    )


if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    args = argument_parsing.parse_args(start_server)
    start_server(**args)
