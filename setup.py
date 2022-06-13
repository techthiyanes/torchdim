from setuptools import setup
import os.path
import os.path
from torch.utils.cpp_extension import (
      CppExtension,
      BuildExtension
)

from subprocess import run
import glob

build_functorch = False

srcs = [
    'dim/csrc/dim.cpp',
]

extra_libraries=[]

if build_functorch:
    cwd = 'third_party/functorch'
    if '#if 0' not in open('third_party/functorch/functorch/csrc/init.cpp', 'r').read():
        print("PATCHING FUNCTORCH")
        run(['git', 'apply', '../../functorch.diff'], cwd=cwd)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ft_home = os.path.join(this_dir, "third_party", "functorch")
    extensions_dir = os.path.join(ft_home, "functorch", "csrc")
    extension_sources = set(
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    )
    srcs.extend(extension_sources)
    print(srcs)
else:
    import functorch._C
    ft_home = os.path.dirname(os.path.dirname(os.path.abspath(functorch.__file__)))
    extra_libraries.append(functorch._C.__file__)

mintorch_C = CppExtension(
      'dim._C',
      srcs,
      include_dirs = [os.path.dirname(os.path.abspath(__file__)), ft_home],
      extra_compile_args = { "cxx": ["-Wno-write-strings", "-Wno-sign-compare"] },
      extra_link_args = extra_libraries
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
      run(['ninja', '-C', 'build/temp.linux-x86_64-3.8', '-t', 'compdb'], stdout=cc)
