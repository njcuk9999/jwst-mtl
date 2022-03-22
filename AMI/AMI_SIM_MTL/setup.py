from setuptools import setup

# Cannot just use True for scm because we are in git subdir
setup(use_scm_version={"root": "../../", "relative_to": __file__})
