from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from Cython.Distutils import build_ext

cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

setup(
  name = 'Demos',
  ext_modules=[
    Extension("composition",
              sources=["composition.pyx", "composition_cpp.cpp"], # Note, you can link against a c++ library instead of including the source
              libraries=["fst", "dl"],
              language="c++"),
    ],
  cmdclass = {'build_ext': build_ext},

)

# python setup.py build_ext -i