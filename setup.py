"""Standard configuration file to create pypi/pip/easy_install compatible packages."""

import subprocess
import sys
import textwrap
from datetime import datetime
from distutils.cmd import Command as DUCommand
from distutils.errors import DistutilsError
from distutils.text_file import TextFile
from os import environ, path

from pkg_resources import add_activation_listener, get_distribution, normalize_path, require, working_set
from setuptools import Command, find_packages, setup
from setuptools.dist import Distribution

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")


AUTHOR = 'Bruno Papa'
AUTHOR_EMAIL = "bruno.papa@studenti.unitn.it"
AUTHOR_COPYRIGHT = '2019 ' + AUTHOR

PROJECT_DESC = 'Python package to simply perform detection in images or tracking in videos'
PROJECT_LONG_DESC = """This is a simple python package."""

PROJECT_URL = 'https://'

REQUIREMENTS = dict(
    # Actual package requirements
    install=[
        'numpy>=1.10,<2',
        'scikit-learn',
        'joblib',
        'opencv-python>=4.1.1',
    ],
    # Setup requirements
    setup=[
    ],
    # Documentation requirements
    doc=[
        # 'sphinx==1.7.2',
    ],
    # Test requirements
    test=[
        'pycodestyle==2.4.0',
        'pylint>=2.0.0',
        'pytest==3.5.0',
        'pytest-mock==1.7.1',
        'pytest-cov==2.5.1',
        'pytest-pylint==0.11.0',
    ],
    # Requirements for examples
    examples=[
        # 'PyQt5<5.11',
        # 'matplotlib',
        # 'scipy',
        # 'tqdm',
        # 'tables',
        # 'scikit-learn',
        # 'colorama',
        # 'h5py',
        # 'natsort',
        # 'pandas',
        # 'joblib',
        # 'xlrd',
        # 'keras',
        # 'tensorflow',
        # 'setuptools<=39.1.0',
    ],
)

# pick from https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: Other/Proprietary License
Natural Language :: English
Operating System :: Unix
Programming Language :: C
Programming Language :: Python :: 3.5
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Software Development
"""


# Add custom entry points here. The default entry point (if configured) is defined in the non-editable section below.
ADDITIONAL_ENTRY_POINTS = [
    # 'airsound-entrypoint=airsound.__main__:somefunc'
]

PROJECT_NAME = 'motPapa'
PACKAGE_NAME = 'motPapa'
MODULE_NAME = 'motPapa'


DEFAULT_ENTRY_POINT = '{}={}.__main__:main'.format(PACKAGE_NAME, MODULE_NAME)
CONSOLE_SCRIPTS = []

ENTRY_POINTS = {
    'console_scripts': CONSOLE_SCRIPTS + ADDITIONAL_ENTRY_POINTS
}

# Version number follows SemVer rules
# See http://semver.org/ for Semantic Versioning 2.0 specs
MAJOR = 0
MINOR = 1
PATCH = 0
SUFFIX = ''
VERSION = '%d.%d.%d%s' % (MAJOR, MINOR, PATCH, SUFFIX)


def setup_package():
    """Configures package setup and checks CLI arguments.

    Returns:
        exit_code (int): 0 if everything worked, 1 if some errors occurred.
    """

    # Raise errors for unsupported commands, improve help output, etc.
    if not check_commands():
        exit_code = 1
        return exit_code

    # Load various requirements from file
    install_requires = REQUIREMENTS['install']
    setup_requires = REQUIREMENTS['setup']
    test_requires = REQUIREMENTS['test']
    doc_requires = REQUIREMENTS['doc']

    # Set extras and add custom requirements found in global map
    extras_require = REQUIREMENTS.copy()
    extras_require['linting'] = setup_requires + test_requires
    extras_require['build'] = setup_requires
    extras_require['doc'] = doc_requires
    extras_require['test'] = install_requires + setup_requires + test_requires
    extras_require['deploy'] = install_requires + setup_requires + doc_requires

    # Retrieve actual list of packages
    package_list = find_packages(exclude=['docs*', 'examples*', 'tests*'])

    metadata = dict(
        author_email=AUTHOR_EMAIL,
        author=AUTHOR,
        classifiers=[c for c in CLASSIFIERS.split('\n') if c],
        description=PROJECT_DESC,
        distclass=PackageDistribution,
        install_requires=install_requires,
        license='Proprietary',
        long_description=PROJECT_LONG_DESC,
        maintainer_email=AUTHOR_EMAIL,
        maintainer=AUTHOR,
        name=PACKAGE_NAME,
        packages=package_list,
        platforms=["Linux"],
        setup_requires=setup_requires,
        tests_require=test_requires,
        url=PROJECT_URL,
        version=VERSION,
        command_options={
            'build_sphinx': {
                'project': ('setup.py', PROJECT_NAME),
                'version': ('setup.py', VERSION),
                'release': ('setup.py', VERSION),
                'copyright': ('setup.py', AUTHOR_COPYRIGHT),
            }
        },
        entry_points=ENTRY_POINTS,
        extras_require=extras_require,
        include_package_data=True,
    )

    exit_code = 0
    try:
        setup(**metadata)
    except SystemExit as inst:
        exit_code = inst.code
    except:  # pylint: disable=W0702
        import traceback
        traceback.print_exc()
        exit_code = 1

    return exit_code


def check_commands():
    """Check the commands and respond appropriately.  Disable broken commands.
    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    if len(sys.argv) < 2:
        # User forgot to give an argument probably, let setuptools handle that
        return True

    '''
    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--obsoletes']
    # Add commands that do more than print info, but also don't need Cython and template parsing
    info_commands.extend(['egg_info', 'install_egg_info', 'rotate'])

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work fine as they are, but are usually used
    # together with one of the commands below and not standalone.  Hence they're not added to good_commands
    good_commands = ('bdist',
                     'build_sphinx',
                     'build',
                     'build_ext',
                     'develop',
                     'sdist',
                     'test', )
    '''

    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = {
        '--requires': 'setup.py --requires is not supported',
        'clean': textwrap.dedent("""
                                 `setup.py clean` is not supported, use one of the following instead:
                                   - `git clean -xdf` (cleans all files)
                                   - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                 files that aren't checked into the git repo)
                                 """),
        'flake8': textwrap.dedent("""
                                  `setup.py flake8` is not supported, use one pylint instead:
                                    - `python3 setup.py test --pylint`
                                  """),
        'pep8': textwrap.dedent("""
                                `setup.py pep8` is not supported, use one pep8 instead:
                                  - `python3 setup.py test --pep8`
                                """),
        'nosetests': textwrap.dedent("""
                                     `setup.py nosetests` is not supported.  Use one of the following
                                     instead:
                                      - `python3 setup.py test`
                                     """),
        'upload': textwrap.dedent("""
                                  `setup.py upload` is not supported, because it's insecure.
                                  Instead, build what you want to upload and upload those files
                                  with `twine upload -s <filenames>` instead.
                                  """),
    }
    for command in ('bdist_dumb', 'bdist_mpkg', 'bdist_msi', 'bdist_rpm' 'bdist_wheel', 'bdist_wininst', 'build_clib',
                    'build_ext', 'build_py', 'build_scripts', 'check', 'easy_install', 'install_data',
                    'install_headers', 'install_lib', 'install_scripts', 'register', 'upload_docs', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands:
        if command in sys.argv[1:]:
            print(textwrap.dedent(bad_commands[command]))
            return False

    return True


class PackageDistribution(Distribution):
    """Helper class to add the ability to set am extra argument in setup():
      - protofiles : Protocol buffer definitions that need compiling
    Also, the class sets the build_py, develop, and test cmdclass options to peculiar command.
    """

    def __init__(self, attrs=None):
        if attrs is None:
            attrs = dict()

        cmdclass = attrs.get('cmdclass', dict())
        cmdclass['build_sphinx'] = BuildDocCommand
        cmdclass['test'] = RunTestCommand
        attrs['cmdclass'] = cmdclass

        # call parent __init__ in old style class
        Distribution.__init__(self, attrs=attrs)


try:
    from sphinx.setup_command import BuildDoc
except:  # pylint: disable=W0702
    class BuildDoc(DUCommand):
        """Fake BuildDoc Command to be used when Sphinx is not installed."""
        user_options = list()

        def finalize_options(self):
            pass

        def initialize_options(self):
            self.source_dir = None
            self.build_dir = None
            self.project = ''
            self.version = ''
            self.release = ''
            self.all_files = False
            self.copyright = ''

        def run(self):
            raise ImportError("Please install Sphinx to build the documentation!")


class BuildDocCommand(BuildDoc):
    """Custom build command to generate documentation."""

    def run(self):
        # Ensure metadata is up-to-date
        self.reinitialize_command('build_py')
        self.run_command('build_py')

        # Build extensions
        self.reinitialize_command('egg_info')
        self.run_command('egg_info')

        self.reinitialize_command('build_ext')
        self.run_command('build_ext')

        ei_cmd = self.get_finalized_command("egg_info")

        # Package was just built...
        package_path = normalize_path(ei_cmd.egg_base)

        old_path = sys.path[:]
        try:
            # Make sure the package is imported from the build dir
            local_path = path.dirname(path.abspath(__file__))
            if local_path in sys.path:
                sys.path.remove(local_path)
            sys.path.insert(0, package_path)

            super().run()
        finally:
            sys.path[:] = old_path


class RunTestCommand(Command):
    # pylint: disable=too-many-instance-attributes
    """Command to run pytest unit tests after build in temporary dir

    The command execute the ``runtests.py`` script for executing the unittests,
    pep8 and pylint tests.

    Example usage::

        # Execute pytest unittests (default)
        python3 setup.py test
        python3 setup.py test --unittests

        # Execute pytest unittests passing custom argumnets to pytest
        python3 setup.py test --unittests --pytest-args="--maxfail=1 tests/pipeline"

        # Execute pep8 validation
        python3 setup.py test --pep8

        # Execute pylint validation
        python3 setup.py test --pylint

        # Show pylint report
        python3 setup.py test --pylintreport

        # Execute all the tests and validations
        python3 setup.py test --pylint --pep8 --unittests
    """

    description = "run package tests using pytest"

    user_options = [
        ('fixtures', None, "show available fixtures"),
        ('no-coverage', None, "disable code coverage report"),
        ('pep8', None, "execute pep8 validation"),
        ('pylint', None, "execute pylint validation"),
        ('pylintreport', None, "show pylint report"),
        ('pytest-args=', None, "extra arguments for pytest"),
        ('skip-build', None, "skip package build"),
        ('unittests', None, "run unittests"),
    ]

    boolean_options = ['fixtures', 'no-coverage', 'pep8', 'pylint', 'pylintreport', 'skip-build', 'unittests']

    def initialize_options(self):
        # pylint: disable=attribute-defined-outside-init
        """Set default values for all the options that this command supports.  Note that these defaults may be
        overridden by other commands, by the setup script, by config files, or by the command-line.  Thus, this is not
        the place to code dependencies between options; generally, 'initialize_options()' implementations are just a
        bunch of "self.foo = None" assignments.
        """

        self.fixtures = False
        self.no_coverage = False
        self.pep8 = False
        self.pylint = False
        self.pylintreport = False
        self.pytest_args = ''
        self.skip_build = False
        self.unittests = False

    def finalize_options(self):
        # pylint: disable=attribute-defined-outside-init
        """Set final values for all the options that this command supports. This is always called as late as possible,
        ie.  after any option assignments from the command-line or from other commands have been done.  Thus, this is
        the place to code option dependencies: if 'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as
        long as 'foo' still has the same value it was assigned in 'initialize_options()'.
        """

        if not any([self.pep8, self.pylint, self.pylintreport, self.unittests, self.fixtures]):
            # Default to unittests
            self.unittests = True

    def run(self):
        """A command's raison d'etre: carry out the action it exists to perform, controlled by the options initialized
        in 'initialize_options()', customized by other commands, the setup script, the command-line, and config files,
        and finalized in 'finalize_options()'.  All terminal output and filesystem interaction should be done by
        'run()'.
        """

        if not self.skip_build:
            # Make sure install requirements are actually installed
            if self.distribution.install_requires:
                self.announce('Installing *install_requires* packages')
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '--user', '-q'] + self.distribution.install_requires
                )

            # Ensure metadata is up-to-date
            self.reinitialize_command('build_py', inplace=0)
            self.run_command('build_py')
            bpy_cmd = self.get_finalized_command("build_py")
            build_path = normalize_path(bpy_cmd.build_lib)

            # Build extensions
            self.reinitialize_command('egg_info', egg_base=build_path)
            self.run_command('egg_info')

            self.reinitialize_command('build_ext', inplace=0)
            self.run_command('build_ext')

        # Make sure test requirements are actually installed
        if self.distribution.tests_require:
            self.announce('Installing *test_require* packages')
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--user', '-q'] + self.distribution.tests_require
            )

        ei_cmd = self.get_finalized_command("egg_info")

        # Get actual package location
        if self.skip_build:
            # Retrieve installation directory
            package_path = normalize_path(get_distribution(PACKAGE_NAME).location)
        else:
            # Package was just built...
            package_path = normalize_path(ei_cmd.egg_base)

        old_path = sys.path[:]
        old_modules = sys.modules.copy()

        try:
            if self.skip_build:
                sys.path.pop(0)
            else:
                sys.path.insert(0, package_path)
            working_set.__init__()
            add_activation_listener(lambda dist: dist.activate())
            require('%s==%s' % (ei_cmd.egg_name, ei_cmd.egg_version))

            if not self._run_tests(package_path):
                raise DistutilsError("Tests failed!")

        finally:
            sys.path[:] = old_path
            sys.modules.clear()
            sys.modules.update(old_modules)
            working_set.__init__()

    def _run_tests(self, package_base_path):
        msg = "Running tests from folder {}".format(package_base_path)
        print('\n\n' + '-' * len(msg) + '\n' + msg + '\n' + '-' * len(msg) + '\n')

        success = True
        pkg_name = PACKAGE_NAME.replace('-', '_')
        package_path = path.join(package_base_path, pkg_name)

        # Possibly run code checks with PEP8
        if self.pep8:
            success &= _run_pep8(package_path)

        # Possibly run code checks with PyLint
        if self.pylint:
            success &= _run_pylint(package_path)

        # Possibly generate code checks report with PyLint
        if self.pylintreport:
            success &= _create_pylint_report(package_path)

        # Possibly provide a list of available fixtures
        if self.fixtures:
            success &= _show_fixtures(self.pytest_args)

        if self.unittests:
            success &= _run_unittests(package_path, self.pytest_args, self.no_coverage)

        return success


def _create_pylint_report(package_path):
    from pylint.lint import Run

    runner = Run(["--rcfile=./.pylintrc", "-r", "y", "--output-format=text", package_path, "--exit-zero"], do_exit=False)
    res = runner.linter.msg_status

    print("Pylintreport result:", res)
    # NOTE: pylint report is not intended to fail
    return True


def _run_pep8(package_path):
    import pycodestyle

    pep8style = pycodestyle.StyleGuide(config_file='./setup.cfg')
    report = pep8style.check_files(paths=[package_path, 'tests', 'experiments'])
    report.print_statistics()
    res = report.total_errors

    print("PEP8 result:", res)
    return res == 0


def _run_pylint(package_path):
    from pylint.lint import Run

    runner = Run(["--rcfile=./.pylintrc", "-E", package_path, "--exit-zero"], do_exit=False)
    res = runner.linter.msg_status

    print("Pylint result:", res)
    return res == 0


def _run_unittests(package_path, pytest_args, no_coverage):
    import pytest

    test_args = list()
    if pytest_args is not None:
        test_args.extend(pytest_args.split())

    if not no_coverage:
        test_args.extend([
            "--cov={}".format(package_path),
            "--cov-config", ".coveragerc",
            "--cov-report", "term-missing:skip-covered",
        ])

    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]  # PyTest reads sys.argv
    try:
        res = pytest.main(test_args)
    except:  # pylint: disable=W0702
        res = 1
    finally:
        sys.argv[:] = old_argv

    print("Unit tests result:", res)
    return res == 0


def _show_fixtures(pytest_args):
    import pytest

    test_args = ['--fixtures']
    if pytest_args is not None:
        test_args.extend(pytest_args.split())

    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]  # PyTest reads sys.argv
    try:
        pytest.main(test_args)
    finally:
        sys.argv[:] = old_argv

    # NOTE: `pytest --fixtures` is not intended to fail
    return True


if __name__ == '__main__':
    setup_exit_code = setup_package()
    exit(setup_exit_code)
