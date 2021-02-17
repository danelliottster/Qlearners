import setuptools

exec(open('Qlearners/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Qlearners",
    version=__version__,
    author="D.L. Elliott",
    author_email="danelliottster@gmail.com",
    description="A package for Q learning (and friends)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danelliottster/Qlearners",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research" ,
        "Programming Language :: Python :: 3" ,
        "License :: OSI Approved :: MIT License" ,
        "Development Status :: 3 - Alpha" ,
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    python_requires='>=3.5',
)
