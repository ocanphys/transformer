import logging
import os
import torch
from pathlib import Path
import numpy as np
import requests
from itertools import islice
from tqdm import tqdm
from tokenizer import Tokenizer


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for notebooks / scripts.

    Idempotent: calling multiple times only updates the level. Safe to call
    at the top of every notebook cell that may be re-run.

    Args:
        level: Log level (e.g. logging.INFO, logging.DEBUG).
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    root.setLevel(level)


def download_and_concat(urls: list[str], output_path: str, separator: str = "\n") -> Path:
    """
    Download text files from URLs and concatenate them into a single file.

    Args:
        urls: List of URLs pointing to plain text files.
        output_path: Path (including filename) where the combined file will be saved.
        separator: String inserted between files (default: newline).

    Returns:
        Path object of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for i, url in enumerate(urls):
            logger.info("[%d/%d] downloading %s", i + 1, len(urls), url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            if i > 0:
                f.write(separator)
            f.write(r.text)

    logger.info("wrote %s (%s bytes)", out, f"{out.stat().st_size:,}")



def textfile_to_tokens_as_binary(source_text,binary_target, tokenizer: Tokenizer, binary_file_mode= "wb"):
    """
    converts a text file into a raw binary file that can be used as memmap
    for training - we are using uint16 which supports vocab size 2^16 max
    source = "data/combined.txt"
    target = "data/train.bin"
    textfile_to_tokens_as_binary(source_text=source, binary_target=target)
    """
    def batched(it, n):
        it = iter(it)
        while batch := list(islice(it, n)):
            yield batch

    total_bytes = os.path.getsize(source_text)
    # iterable that returns lines from the text file
    with open(source_text, "r") as source_file, \
         tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"tokenizing {source_text}") as pbar:

        def tracked_lines():
            for line in source_file:
                pbar.update(len(line.encode("utf-8")))
                yield line

        # encode line by line and returns an iterator (lazy) of tokens
        token_stream = tokenizer.encode_iterable(tracked_lines())
        with open(binary_target, binary_file_mode) as target_file:
            for chunk in batched(token_stream, 1 << 20):
                np.array(chunk, dtype=np.uint16).tofile(target_file)


def get_batch(data: np.ndarray, batch_size: int, context_length: int,
              device: torch.device | str = 'cpu'):
    high = len(data) - context_length
    starts = np.random.randint(0, high, size=batch_size)
    # cast on the numpy side - Embedding needs long
    inputs  = np.stack([data[i:i+context_length]     for i in starts]).astype(np.int64)
    targets = np.stack([data[i+1:i+context_length+1] for i in starts]).astype(np.int64)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def save_checkpoint(model, optimizer, iteration, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


class LiveLossPlot:
    """Context manager that draws a live loss curve in a Jupyter notebook.

    Usage:
        with LiveLossPlot(every=10) as plot:
            for iteration in range(n_steps):
                ...
                plot.log(loss.item(), iteration)

    Requires `%matplotlib widget` (ipympl) in the notebook for live updates.
    """

    def __init__(self, every: int = 10, figsize: tuple[int, int] = (10, 4),
                 title: str = "Training loss (live)", text_color: str = "white"):
        self.every = every
        self.figsize = figsize
        self.title = title
        self.text_color = text_color
        self.iterations: list[int] = []
        self.losses: list[float] = []
        self.val_steps: list[int] = []
        self.val_losses: list[float] = []

    def __enter__(self):
        import matplotlib.pyplot as plt
        from IPython.display import display

        # ioff prevents auto-display so we only render via the display handle
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # transparent background + text_color for ticks/labels to match a dark (VS Code) theme
        self.fig.patch.set_alpha(0)
        self.ax.patch.set_alpha(0)
        for spine in self.ax.spines.values():
            spine.set_color(self.text_color)
        self.ax.tick_params(colors=self.text_color)
        self.ax.xaxis.label.set_color(self.text_color)
        self.ax.yaxis.label.set_color(self.text_color)
        self.ax.title.set_color(self.text_color)

        (self.line,) = self.ax.plot([], [], label="train")
        (self.val_line,) = self.ax.plot([], [], color="tab:orange", marker="o", label="val")
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("loss")
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3, color=self.text_color)
        legend = self.ax.legend(loc="upper right")
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_color(self.text_color)
        self._dh = display(self.fig, display_id=True)  # reserve an output slot we can overwrite
        return self

    def log(self, loss: float, iteration: int) -> None:
        self.iterations.append(iteration)
        self.losses.append(loss)
        if (len(self.losses) - 1) % self.every == 0:
            self._redraw()

    def log_val(self, loss: float, iteration: int) -> None:
        """Record a validation loss at the given training iteration."""
        self.val_steps.append(iteration)
        self.val_losses.append(loss)
        self._redraw()

    def _redraw(self) -> None:
        self.line.set_data(self.iterations, self.losses)
        self.val_line.set_data(self.val_steps, self.val_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self._dh.update(self.fig)

    def __exit__(self, exc_type, exc_val, exc_tb):
        import matplotlib.pyplot as plt

        del exc_type, exc_val, exc_tb
        self._redraw()       # final paint so the last few steps show up
        plt.close(self.fig)  # prevent a duplicate render at cell end
        return False


class LMDataLoader(torch.utils.data.IterableDataset):
    """Infinite random-batch loader over a uint16 token memmap."""

    def __init__(self, path: str, batch_size: int, context_length: int,
                 device: torch.device | str = "cpu"):
        self.path = path
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def __iter__(self):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        while True:
            yield get_batch(data, self.batch_size, self.context_length, self.device)