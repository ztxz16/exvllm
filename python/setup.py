from setuptools import setup, find_packages

all_require = ['vllm']

setup (
    name = "exvllm",
    version = "0.0.0.3",
    author = "huangyuyang",
    author_email = "ztxz16@foxmail.com",
    description = "extend for vllm",
    url = "https://github.com/ztxz16/exvllm",
    entry_points = {
        'console_scripts' : [
            'exvllm=exvllm.cli:main'
        ]
    },
    packages = ['exvllm', 'exvllm/vllm'],
    package_data = {
        '': ['*.dll', '*.so', '*.dylib', '*.so.*']
    },
    install_requires = [] + all_require,
    extras_require = {
        'all': all_require
    },
)
