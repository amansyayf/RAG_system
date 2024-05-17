from setuptools import find_packages, setup

setup(
    name='QAsystem',
    version='0.0.1',
    author='aman',
    author_email='asysyfetdinov@gmail.com',
    packages=find_packages(),
    install_requires=['langchain', 'langchainhub', 'bs4', 'tiktoken', 'openai', 'langchain_community', 'chromadb', 'awscli', 'stramlit', 'pypdf', 'faiss-cpu']
)