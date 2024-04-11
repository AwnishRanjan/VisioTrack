from setuptools import setup , find_packages 
from  typing import List
Hyper = '-e .'

def get_requirements(file_path :str)->List[str]:
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [i.replace('\n','')for i in requirements]
        if Hyper in requirements:
            requirements.remove(Hyper)

    return requirements




setup(
    name='Face Recognition',
    version='0.0.1',
    author='Awnish Ranjan',
    author_email='ranjanawnish07@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages=find_packages()

)