import numpy

from aspectizing import any_callable, aspect, woody_logger

aspect(numpy, any_callable, "", woody_logger, dry_run=True)
