from setuptools import setup

setup(name='pyLRT',
      version='0.2',
      author='Edward Gryspeerdt',
      author_email='e.gryspeerdt@imperial.ac.uk',
      maintainer='Edward Gryspeerdt',
      maintainer_email='e.gryspeerdt@imperial.ac.uk',
      description='A simple python interface/wrapper for LibRadTran',
      license='BSD (3-clause)',
      url='https://github.com/EdGrrr/pyLRT',
      install_requires=['xarray',
                        'numpy',
                        'scipy',
                        'netCDF4',
                        'matplotlib'],
      packages=['pyLRT'],)
