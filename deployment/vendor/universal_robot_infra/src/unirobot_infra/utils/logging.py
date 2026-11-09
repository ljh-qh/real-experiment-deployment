import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs, stacklevel=2)


logging.Logger.success = success


class ProjectFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        short_name = record.name.split(".")[-1]
        record.msg = f"[{short_name}] {record.getMessage()}"
        record.args = ()
        return super().format(record)


class ProjectRichHandler(RichHandler):
    def get_level_text(self, record: logging.LogRecord) -> Text:
        if record.levelno == SUCCESS_LEVEL:
            return Text.styled(record.levelname.ljust(8), "bold green")
        return super().get_level_text(record)


def setup_logging(level: str = "INFO", console: Console | None = None) -> None:
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
    rich_console = console if console is not None else Console(stderr=True)

    package_logger = logging.getLogger("unirobot_infra")
    for handler in package_logger.handlers:
        if isinstance(handler, ProjectRichHandler):
            handler.console = rich_console
            package_logger.setLevel(level)
            handler.setLevel(level)
            return

    handler = ProjectRichHandler(
        console=rich_console,
        rich_tracebacks=True,
        show_path=False,
        markup=False,
    )
    handler.setLevel(level)
    handler.setFormatter(ProjectFormatter("%(message)s"))

    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    package_logger = logging.getLogger("unirobot_infra")
    if not any(isinstance(handler, ProjectRichHandler) for handler in package_logger.handlers):
        setup_logging()
    if name.startswith("unirobot_infra"):
        return logging.getLogger(name)
    return logging.getLogger(f"unirobot_infra.{name}")
