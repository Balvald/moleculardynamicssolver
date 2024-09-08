lib_sources = [  # All source files (excluding headers)
    'hello.cpp', 'verlet.cpp', 'xyz.cpp', 'lj_direct_summation.cpp', 'berendsen.cpp',
    'neighbors.cpp', 'lj.cpp', 'atoms.cpp', 'ducastelle.cpp', 'domain.cpp', 'ih.C'
]

lib = static_library(
    'lib',  # Unique name
    lib_sources,
    dependencies : [eigen, mpi]
)

lib_incdirs = include_directories('.')
