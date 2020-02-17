'''
MatProd - Fast repeated multiplication of 2x2 matrices
Copyright (C) 2020 Christopher M. Pierce (contact@chris-pierce.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import setuptools
import numpy as np

# Load my long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Add in my c-extension
ext_modules = [setuptools.Extension('matprod', sources = ['matprod.c'],
    include_dirs = [np.get_include()]) ]

# Run the setup command
setuptools.setup(
        name = 'matprod',
        version = '1.0',
        description = 'Fast repeated multiplication of 2x2 matrices in a compiled numpy extension',
        author = 'Christopher M. Pierce',
        author_email = 'contact@chris-pierce.com',
        python_requires='>=3.1',
        long_description=long_description,
        ext_modules = ext_modules,
        license = 'GNU Affero General Public License v3 or later (AGPLv3+)',
        url = 'https://github.com/electronsandstuff/MatProd',
        setup_requires = [
            'numpy',
            'wheel',
        ],
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
            "Development Status :: 4 - Beta",
            "Operating System :: OS Independent",
        ],
    )
