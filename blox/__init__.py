from . import dataset
from . import search_space
from . import graph_utils
from . import utils

from .utils import add_module_properties
from .utils import staticproperty


Dataset = dataset.Dataset
BenchmarkingDataset = dataset.BenchmarkingDataset
StaticInfoDataset = dataset.StaticInfoDataset
EnvInfoDataset = dataset.EnvInfoDataset

from_folder = dataset.from_folder


def _get_version():
    from . import version
    return version.version

def _get_has_repo():
    from . import version
    return version.has_repo

def _get_repo():
    from . import version
    return version.repo

def _get_commit():
    from . import version
    return version.commit


add_module_properties(__name__, {
    '__version__': staticproperty(staticmethod(_get_version)),
    '__has_repo__': staticproperty(staticmethod(_get_has_repo)),
    '__repo__': staticproperty(staticmethod(_get_repo)),
    '__commit__': staticproperty(staticmethod(_get_commit))
})
