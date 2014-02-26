from distutils.core import setup, Extension

module1 = Extension('ParODE',
                    sources = ['ParODEmodule.c'],
		    include_dirs = ['/home/alexandru/Projects/ParallelODESolver/Software/Cpp/Include'],
		    define_macros = [('C_INCLUDE', 1), 
					('KERNEL_FOLDER', '\"/home/alexandru/Projects/ParallelODESolver/Software/Binary/\"'), 
					('KERNEL_INCLUDE_FOLDER', '\"/home/alexandru/Projects/ParallelODESolver/Software/Binary\"')],
		    libraries = ['ParODEShared'],
		    library_dirs = ['/home/alexandru/Projects/ParallelODESolver/Software/Binary'],
		    runtime_library_dirs = ['/home/alexandru/Projects/ParallelODESolver/Software/Binary'])

setup (name = 'ParODE',
       version = '1.0',
       description = 'Small package that implements GPU accelerated LTI simulation',
       ext_modules = [module1])

