from setuptools import setup, find_packages

setup(
    name='gym_tiadas',
    version='0.0.1',
    install_requires=['gym', 'numpy'],  # And any other dependencies foo needs
    packages=[
        package for package in find_packages() if package.startswith('gym')
    ]
)
