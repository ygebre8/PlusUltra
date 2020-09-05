from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["Propagate.pyx", "Dipole_Acceleration_Matrix.pyx", "Field_Free.pyx", "Interaction.pyx", "Coefficent_Calculator.pyx"])
)