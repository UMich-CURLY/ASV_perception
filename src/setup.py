from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'asv_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yui C.',
    maintainer_email='yuic@umich.edu',
    description='ASV Perception',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obj_seg = sensor_frontend.objseg_node:main',
            'pcd_fil = sensor_frontend.pcdFilter_node:main',
            'pcd_pub = semantic_mapping.ConvBKI.semantic_pcd_publisher:main',
            'pcd_pub_global = semantic_mapping.ConvBKI.semantic_pcd_publisher_global:main',
        ],
    },
)