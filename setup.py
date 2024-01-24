import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
HEADER = [ROOT_DIR]
SOURCES = [
    os.path.join(ROOT_DIR, 'knn_aggregate.cpp'),
    os.path.join(ROOT_DIR, 'knn_aggregate_kernel.cu'),
]
#SOURCES_ = glob.glob(os.path.join(ROOT_DIR, '*.cpp')) #+ glob.glob(os.path.join(ROOT_DIR, '*.cu'))
print(SOURCES)

setup(
    name='cuda_knn_aggregate',
    version='0.0',
    author='S.-Y S',
    description='cuda_knn_aggregate',
    long_description='Direct kNN feature aggregation',
    packages=['cuda_knn_aggregate'],
    ext_modules=[
        CUDAExtension(
            name='cuda_knn_aggregate._C',
            sources=SOURCES,
            include_dirs=HEADER,
            extra_compile_args={
                'cxx': ['-O2'], 
                'nvcc': ['-O2', '-v'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)