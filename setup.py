from setuptools import setup

setup(name='tmiscpy',
      version='0.1',
      description='Miscellaneous Python function for Tim Mak',
      url='http://github.com/tshmak/tmispy',
      author='Tim Mak',
      author_email='tim.sh.mak@gmail.com',
      license='MIT',
      packages=['tmiscpy'],
      install_requires=['jieba', 'tabulate', 'numpy', 'scipy', 
                        'matplotlib', 'subprocess'],
      include_package_data=True,
      zip_safe=False)

