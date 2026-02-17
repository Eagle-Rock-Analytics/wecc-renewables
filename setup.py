from setuptools import find_packages, setup

setup(
    name="renewable-profiles",
    version="0.1.0",
    description="ERA development of renewable energy profiles",
    author="Beth McClenny, Héctor A. Inda-Díaz, Neil Schroeder, Nicole Keeney",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
