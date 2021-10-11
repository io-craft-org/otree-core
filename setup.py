import os
import sys
from setuptools import setup, find_packages
import shutil
from pathlib import Path

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

import otree

version = otree.__version__


MSG_PY_VERSION = """
Error: This version of oTree requires Python 3.7 or higher.
"""

if sys.version_info < (3, 7):
    sys.exit(MSG_PY_VERSION)


def clean_requirements(requirements_text):
    required_raw = requirements_text.splitlines()
    required = []
    for line in required_raw:
        req = line.split('#')[0].strip()
        if req:
            required.append(req)
    return required


README = Path('README.rst').read_text('utf8')
required = clean_requirements(Path('requirements.txt').read_text())
required_mturk = clean_requirements(Path('requirements_mturk.txt').read_text())


if sys.argv[-1] == 'publish':

    if Path('dist').is_dir():
        shutil.rmtree('dist')
    for cmd in [
        "python setup.py sdist bdist_wheel",
        "twine upload dist/*",
        f'git tag -a {version} -m "version {version}"',
        "git push --tags",
    ]:
        sys.stdout.write(cmd + '\n')
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise AssertionError
    if Path('build').is_dir():
        shutil.rmtree('build')

    sys.exit()


setup(
    name='otree',
    version=version,
    include_package_data=True,
    license='MIT License',
    # 2017-10-03: find_packages function works correctly, but tests
    # are still being included in the package.
    # not sure why. so instead i use
    # recursive-exclude in MANIFEST.in.
    packages=find_packages(),
    description=(
        'oTree is a toolset that makes it easy to create and '
        'administer web-based social science experiments.'
    ),
    long_description=README,
    url='http://otree.org/',
    author='chris@otree.org',
    author_email='chris@otree.org',
    install_requires=required,
    entry_points={'console_scripts': ['otree=otree.main:execute_from_command_line']},
    zip_safe=False,
    extras_require={'mturk': required_mturk},
)
