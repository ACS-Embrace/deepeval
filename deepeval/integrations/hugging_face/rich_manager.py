from typing import Union

from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.console import Console
from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn


class RichManager:
    def __init__(self, show_table: bool, total_train_epochs: int) -> None:
        """
        Initialize RichManager.

        Args:
            show_table (bool): Flag to show or hide the table.
            total_train_epochs (int): Total number of training epochs.
        """
        self.show_table = show_table
        self.total_train_epochs = total_train_epochs
        self.console = Console()
        self.live = None
        self.train_bar_started = False
        self._ref_count: int = 0  # number of callbacks sharing this manager
        self._last_advanced_epoch: int = 0  # prevents multiple advances per epoch

        self.progress_bar_columns = [
            TextColumn(
                "{task.description} [progress.percentage][green][{task.percentage:>3.1f}%]:",
                justify="right",
            ),
            BarColumn(),
            TextColumn(
                "[green][ {task.completed}/{task.total} epochs ]",
                justify="right",
            ),
        ]
        self.spinner_columns = [
            TextColumn("{task.description}", justify="right"),
            SpinnerColumn(spinner_name="simpleDotsScrolling"),
        ]

        self.empty_column = Text("\n")

    def _initialize_progress_trackers(self) -> None:
        """
        Initialize progress trackers (progress and spinner columns).
        """
        self.progress = Progress(*self.progress_bar_columns, auto_refresh=False)
        self.spinner = Progress(*self.spinner_columns)

        self.progress_task = self.progress.add_task(
            "Train Progress", total=self.total_train_epochs
        )
        self.spinner_task = self.spinner.add_task("Initializing")

        column_list = [self.spinner, self.progress, self.empty_column]
        column_list.insert(0, Table()) if self.show_table else None

        column = Columns(column_list, equal=True, expand=True)
        self.live.update(column, refresh=True)

    def change_spinner_text(self, text: str) -> None:
        """
        Change the text displayed in the spinner.

        Args:
            text (str): Text to be displayed in the spinner.
        """
        if self.live is None or not self.live._started:
            return
        self.spinner.reset(self.spinner_task, description=text)

    def stop(self) -> None:
        """
        Stop the live display.

        When multiple callbacks share this manager, stop() decrements the
        internal reference count and only tears down the Live display when
        the last callback has called stop().
        """
        if self._ref_count > 1:
            self._ref_count -= 1
            return
        if self.live is not None and self.live._started:
            self.live.stop()
        self.live = None
        self._ref_count = 0

    def start(self) -> None:
        """
        Start the live display and initialize progress trackers.

        Idempotent: if the display is already running (i.e. another callback
        sharing this manager already called start()), only the reference count
        is incremented and the display is left untouched.
        """
        self._ref_count += 1
        if self.live is not None and self.live._started:
            return
        self.live = Live(auto_refresh=True, console=self.console)
        self.train_bar_started = False
        self.live.start()
        self._initialize_progress_trackers()

    def update(self, column: Columns) -> None:
        """
        Update the live display with a new column.

        Args:
            column (Columns): New column to be displayed.
        """
        if self.live is None or not self.live._started:
            return
        self.live.update(column, refresh=True)

    def create_column(self) -> Union[Columns, Table]:
        """
        Create a new column with an optional table.

        Returns:
            Tuple[Columns, Table]: Tuple containing the new column and an optional table.
        """
        new_table = Table()

        column_list = [self.spinner, self.progress, self.empty_column]
        column_list.insert(0, new_table) if self.show_table else None

        column = Columns(column_list, equal=True, expand=True)
        return column, new_table

    def advance_progress(self, epoch: float) -> None:
        """Advance the progress tracker, at most once per epoch."""
        epoch_int = int(epoch)
        if epoch_int <= self._last_advanced_epoch:
            return
        self._last_advanced_epoch = epoch_int
        if not self.train_bar_started:
            self.progress.start()
            self.train_bar_started = True
        self.progress.update(self.progress_task, advance=1)
