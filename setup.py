from setuptools import setup
import os.path
import functorch._C
import os.path
from torch.utils.cpp_extension import (
      CppExtension,
      BuildExtension
)

srcs = [
      'dim/csrc/dim.cpp',
]

ft_home = os.path.dirname(os.path.dirname(os.path.abspath(functorch.__file__)))

mintorch_C = CppExtension(
      'dim._C',
      srcs,
      include_dirs = [os.path.dirname(os.path.abspath(__file__)), ft_home],
      extra_compile_args = { "cxx": ["-Wno-write-strings", "-Wno-sign-compare", "-include-pch", "/raid/zdevito/dumptruck/dim/all_headers.h.pch"] },
      extra_link_args = [functorch._C.__file__]
)

setup(name='dim',
      version='1.0',
      description='first class dimensions',
      author='',
      author_email='',
      url='',
      packages=['dim'],
      ext_modules=[mintorch_C],
      cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
     )

with open('compile_commands.json', 'w') as cc:
      from subprocess import run
      run(['ninja', '-C', 'build/temp.linux-x86_64-3.8', '-t', 'compdb'], stdout=cc)
