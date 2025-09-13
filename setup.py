from setuptools import setup, find_packages

def read_requirements(filename: str) -> list[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="rl-square",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.10',
    author="Yanfeng Lu",
    author_email="luyanfeng_nlp@qq.com",
    description="RL2: Ray Less Reinforcement Learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yanfeng98/fork-RL2"
)