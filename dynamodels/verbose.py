"""Verbosity of dynamodels.

The package writes its diagnostic messages (e.g. the start and the closing of the multiprocessing pool of an
ensemble) to the logger ``'dynamodels'`` and its children, which are silent by default. `set_verbose` shows them on
stderr for debugging; an application that configures `logging` itself can set the level of the ``'dynamodels'``
logger instead. ``set_verbose(False)`` returns that level to the default (NOTSET).
"""

import logging

__all__ = ['set_verbose']

_HANDLER = 'dynamodels.verbose'


def set_verbose(on: bool = True) -> None:
    """Show (``on=True``) or silence (``on=False``) the diagnostic messages of dynamodels.

    On: the logger ``'dynamodels'`` takes the level DEBUG and one stream handler on stderr, which prints
    ``[dynamodels] message``; calling it again adds no second handler. Off: the handler is removed and the level
    returns to the default (NOTSET), so that the package is silent again unless the application's own logging
    configuration shows its messages. Off does not restore a level that the application had set on ``'dynamodels'``,
    so an application that sets its own level should set it after ``set_verbose(False)``.
    """
    logger = logging.getLogger('dynamodels')
    handlers = [h for h in logger.handlers if h.get_name() == _HANDLER]
    if on:
        if not handlers:
            handler = logging.StreamHandler()
            handler.set_name(_HANDLER)
            handler.setFormatter(logging.Formatter('[dynamodels] %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    else:
        for handler in handlers:
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
