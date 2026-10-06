from setuptools import find_packages, setup

package_name = 'dofbot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='natu',
    maintainer_email='natu@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "ik_move_above = dofbot_control.ik_move_above:main",
            "collect_imitation_episodes = dofbot_control.collect_imitation_episodes:main",
            "collect_smolvla_pick_episodes = dofbot_control.collect_smolvla_pick_episodes:main",
        ],
    },
)
