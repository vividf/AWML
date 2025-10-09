"""
Events Utils

Modified from Detectron2

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
import sys
import traceback
from collections import defaultdict
from contextlib import contextmanager

__all__ = [
    "EventStorage",
    "ExceptionWriter",
]

_CURRENT_STORAGE_STACK = []


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.
    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(AverageMeter)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

    # def put_image(self, img_name, img_tensor):
    #     """
    #     Add an `img_tensor` associated with `img_name`, to be shown on
    #     tensorboard.
    #     Args:
    #         img_name (str): The name of the image to put into tensorboard.
    #         img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
    #             Tensor of shape `[channel, height, width]` where `channel` is
    #             3. The image format should be RGB. The elements in img_tensor
    #             can either have values in [0, 1] (float32) or [0, 255] (uint8).
    #             The `img_tensor` will be visualized in tensorboard.
    #     """
    #     self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, n=1, smoothing_hint=False):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        history.update(value, n)
        self._latest_scalars[name] = (value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert existing_hint == smoothing_hint, "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    # def put_scalars(self, *, smoothing_hint=True, **kwargs):
    #     """
    #     Put multiple scalars from keyword arguments.
    #     Examples:
    #         storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
    #     """
    #     for k, v in kwargs.items():
    #         self.put_scalar(k, v, smoothing_hint=smoothing_hint)
    #
    # def put_histogram(self, hist_name, hist_tensor, bins=1000):
    #     """
    #     Create a histogram from a tensor.
    #     Args:
    #         hist_name (str): The name of the histogram to put into tensorboard.
    #         hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
    #             into a histogram.
    #         bins (int): Number of histogram bins.
    #     """
    #     ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()
    #
    #     # Create a histogram with PyTorch
    #     hist_counts = torch.histc(hist_tensor, bins=bins)
    #     hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)
    #
    #     # Parameter for the add_histogram_raw function of SummaryWriter
    #     hist_params = dict(
    #         tag=hist_name,
    #         min=ht_min,
    #         max=ht_max,
    #         num=len(hist_tensor),
    #         sum=float(hist_tensor.sum()),
    #         sum_squares=float(torch.sum(hist_tensor**2)),
    #         bucket_limits=hist_edges[1:].tolist(),
    #         bucket_counts=hist_counts.tolist(),
    #         global_step=self._iter,
    #     )
    #     self._histograms.append(hist_params)

    def history(self, name):
        """
        Returns:
            AverageMeter: the history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.
        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,
                itr,
            )
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.
        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []

    def reset_history(self, name):
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        ret.reset()

    def reset_histories(self):
        for name in self._history.keys():
            self._history[name].reset()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count


class ExceptionWriter:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            tb = traceback.format_exception(exc_type, exc_val, exc_tb)
            formatted_tb_str = "".join(tb)
            self.logger.error(formatted_tb_str)
            sys.exit(1)  # This prevents double logging the error to the console
