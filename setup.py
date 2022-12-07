#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py

import sys
import importlib
import importlib.util
from pathlib import Path

package_name = 'blox'

version_file = Path(__file__).parent.joinpath(package_name, 'version.py')
spec = importlib.util.spec_from_file_location('{}.version'.format(package_name), version_file)
package_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_version)
sys.modules[spec.name] = package_version


blox_C = Extension('blox.C', sources=['blox/_C/io_parse.c'])


class build_maybe_inplace(build_py):
    def run(self):
        global package_version
        package_version = importlib.reload(package_version)
        _dist_file = version_file.parent.joinpath('_dist_info.py')
        assert not _dist_file.exists()
        _dist_file.write_text('\n'.join(map(lambda attr_name: attr_name+' = '+repr(getattr(package_version, attr_name)), package_version.__all__)) + '\n')
        ret = super().run()
        _dist_file.unlink()
        return ret


with Path(__file__).parent.joinpath('README.md').open('r') as f:
    long_desc = f.read()


setup(name=package_name,
      version=package_version.version,
      description='Library for the Blox dataset',
      author='SAIC-Cambridge, On-Device Team',
      author_email='l.dudziak@samsung.com',
      url='https://github.com/SamsungLabs/blox',
      download_url='https://github.com/SamsungLabs/blox',
      long_description=long_desc,
      long_description_content_type='text/markdown',
      python_requires='>=3.6.0',
      setup_requires=[
          'GitPython'
      ],
      install_requires=[
          'tqdm',
          'numpy',
          'torch',
          'torchvision',
          'networkx>=2.5',
          'ptflops',
          'tensorboard',
          'pyyaml'
      ],
      dependency_links=[
      ],
      packages=find_packages(where='.', include=[ 'blox', 'blox.*' ]),
      package_dir={ '': '.' },
      data_files=[],
      cmdclass={
          'build_py': build_maybe_inplace
      },
      ext_modules=[blox_C]
)
