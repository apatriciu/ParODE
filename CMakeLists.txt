cmake_minimum_required(VERSION 2.6)
project(ParallelODESolver)

# Auto-select bitness based on platform
if( NOT BITNESS )
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BITNESS 64)
    else()
        set(BITNESS 32)
    endif()
endif()

# Select bitness for non-msvc platform. Can be specified as -DBITNESS=32/64 at command-line
if( NOT MSVC )
    set(BITNESS ${BITNESS} CACHE STRING "Specify bitness")
    set_property(CACHE BITNESS PROPERTY STRINGS "64" "32")
endif()

# Unset OPENCL_LIBRARIES, so that corresponding arch specific libs are found when bitness is changed
unset(OPENCL_LIBRARIES CACHE)
if( BITNESS EQUAL 64 )
    set(BITNESS_SUFFIX x86_64)
elseif( BITNESS EQUAL 32 )
    set(BITNESS_SUFFIX x86)
else()
    message( FATAL_ERROR "Bitness specified is invalid" )
endif()

set(GPU_DEBUGGING ${GPU_DEBUGGING} CACHE STRING "GPU Debugging?")
set_property(CACHE GPU_DEBUGGING PROPERTY STRINGS "Yes" "No")

# Set CMAKE_BUILD_TYPE (default = Release)
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release)
endif()

# Set platform
if( NOT UNIX )
	set(PLATFORM win)
else()
	set(PLATFORM lnx)
endif()

############################################################################
# Find OpenCL include and libs
find_path( OPENCL_INCLUDE_DIRS 
    NAMES OpenCL/cl.h CL/cl.h
    HINTS ../../../../include/ $ENV{AMDAPPSDKROOT}/include/ /usr/local/cuda/include/
)
mark_as_advanced(OPENCL_INCLUDE_DIRS)

find_library( OPENCL_LIBRARIES
	NAMES OpenCL
	HINTS ../../../../lib/ $ENV{AMDAPPSDKROOT}/lib /usr/local/cuda/lib/
	PATH_SUFFIXES ${PLATFORM}${BITNESS} ${BITNESS_SUFFIX}
)
mark_as_advanced( OPENCL_LIBRARIES )

if( OPENCL_INCLUDE_DIRS STREQUAL "" OR OPENCL_LIBRARIES STREQUAL "")
	message( FATAL_ERROR "Could not locate OpenCL include & libs" )
endif( )

# Boost libraries
if( PLATFORM STREQUAL "win" )
	include_directories(C:/Software/boost_1_54_0)
else()
	include_directories(~/Software/boost_1_55_0)
endif()

set( COMPILER_FLAGS " " )
set( LINKER_FLAGS " " )
set( ADDITIONAL_LIBRARIES "" )

file(GLOB INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h" )
include_directories( ${OPENCL_INCLUDE_DIRS} ../../../../include/SDKUtil $ENV{AMDAPPSDKROOT}/include/SDKUtil )

# POS Folders
set(POS_INCLUDE_FOLDER ${CMAKE_SOURCE_DIR}/Include)
set(POS_SOURCE_FOLDER ${CMAKE_SOURCE_DIR}/Sources)
set(POS_TESTSSOURCE_FOLDER ${CMAKE_SOURCE_DIR}/TestsSources)
set(POS_KERNEL_FOLDER ${CMAKE_SOURCE_DIR}/Kernels)

include_directories(${POS_INCLUDE_FOLDER})

set(POS_SOURCE_FILES 	${POS_SOURCE_FOLDER}/AdamsMoulton.cpp
			${POS_SOURCE_FOLDER}/GPUFixedStepIterative.cpp
			${POS_SOURCE_FOLDER}/GPUManagement.cpp
			${POS_SOURCE_FOLDER}/GPUODESolver.cpp
			${POS_SOURCE_FOLDER}/ParODE.cpp
			${POS_SOURCE_FOLDER}/RungeKutta4thOrder.cpp
			${POS_SOURCE_FOLDER}/KernelWrapper.cpp
			${POS_SOURCE_FOLDER}/BTimesURK4thOrder.cpp
			${POS_SOURCE_FOLDER}/DistributeA.cpp
			${POS_SOURCE_FOLDER}/LeftLeafCopyKernelScanUpsweep.cpp
			${POS_SOURCE_FOLDER}/MMKernelScanDownsweep.cpp
			${POS_SOURCE_FOLDER}/MMKernelScanUpsweep.cpp
			${POS_SOURCE_FOLDER}/MVVKernelScanDownsweep.cpp
			${POS_SOURCE_FOLDER}/MVVKernelScanUpsweep.cpp
			${POS_SOURCE_FOLDER}/RootCopyKernelScanDownsweep.cpp
			${POS_SOURCE_FOLDER}/SystemOutput.cpp
			${POS_SOURCE_FOLDER}/BTimesUAdamsMoulton.cpp
			${POS_SOURCE_FOLDER}/ParODEException.cpp
			${POS_INCLUDE_FOLDER}/AdamsMoulton.h
			${POS_INCLUDE_FOLDER}/GPUFixedStepIterative.h
			${POS_INCLUDE_FOLDER}/GPUManagement.h
			${POS_INCLUDE_FOLDER}/GPUODESolver.h
			${POS_INCLUDE_FOLDER}/ParODE.h
			${POS_INCLUDE_FOLDER}/KernelWrapper.h
			${POS_INCLUDE_FOLDER}/RungeKutta4thOrder.h
			${POS_INCLUDE_FOLDER}/BTimesURK4thOrder.h
			${POS_INCLUDE_FOLDER}/DistributeA.h
			${POS_INCLUDE_FOLDER}/LeftLeafCopyKernelScanUpsweep.h
			${POS_INCLUDE_FOLDER}/MMKernelScanDownsweep.h
			${POS_INCLUDE_FOLDER}/MMKernelScanUpsweep.h
			${POS_INCLUDE_FOLDER}/MVVKernelScanDownsweep.h
			${POS_INCLUDE_FOLDER}/MVVKernelScanUpsweep.h
			${POS_INCLUDE_FOLDER}/RootCopyKernelScanDownsweep.h
			${POS_INCLUDE_FOLDER}/SystemOutput.h
			${POS_INCLUDE_FOLDER}/BTimesUAdamsMoulton.h
			${POS_INCLUDE_FOLDER}/ParODEException.h)

set(POS_KERNEL_FILES ${POS_KERNEL_FOLDER}/BTimesURK4thOrder.cl
			${POS_KERNEL_FOLDER}/DistributeA.cl
			${POS_KERNEL_FOLDER}/LeftLeafCopyKernelScanUpsweep.cl
			${POS_KERNEL_FOLDER}/MMKernelScanDownsweep.cl
			${POS_KERNEL_FOLDER}/MMKernelScanUpsweep.cl
			${POS_KERNEL_FOLDER}/MVVKernelScanDownsweep.cl
			${POS_KERNEL_FOLDER}/MVVKernelScanUpsweep.cl
			${POS_KERNEL_FOLDER}/RootCopyKernelScanDownsweep.cl
			${POS_KERNEL_FOLDER}/SystemOutput.cl
			${POS_KERNEL_FOLDER}/BTimesUAdamsMoulton.cl
			${POS_KERNEL_FOLDER}/u.cl)

set(POS_DEBUG_FILES ${POS_INCLUDE_FOLDER}/DebugUtilities.h
		    ${POS_SOURCE_FOLDER}/DebugUtilities.cpp)

# gcc/g++ specific compile options
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set( COMPILER_FLAGS "${COMPILER_FLAGS} -msse2 -Wwrite-strings " )
    if(${GPU_DEBUGGING} STREQUAL "Yes")
	add_definitions(-D_GPU_DEBUUGGING_)
    endif()
    
    # Note: "rt" is not present on mingw
    if( UNIX )
		if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
			set( COMPILER_FLAGS " -g " )
		endif( )
        set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} "rt" )
    endif( )
    
    if( BITNESS EQUAL 32 )
        set( COMPILER_FLAGS "${COMPILER_FLAGS} -m32 " )
        set( LINKER_FLAGS "${LINKER_FLAGS} -m32 " )
    else( )
        set( COMPILER_FLAGS "${COMPILER_FLAGS} -m64 " )
        set( LINKER_FLAGS "${LINKER_FLAGS} -m64 " )
    endif( )
    
    set( COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_GXX} " )
    set( LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_GXX} " )
    set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_GXX} )
elseif( MSVC )
    # Samples can specify additional libs/flags using EXTRA* defines
    add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /nologo" )
    if(${GPU_DEBUGGING} STREQUAL "Yes")
	add_definitions(-D_GPU_DEBUUGGING_)
    endif()

    set( COMPILER_FLAGS "${COMPILER_FLAGS} ${EXTRA_COMPILER_FLAGS_MSVC} " )
    set( LINKER_FLAGS "${LINKER_FLAGS} ${EXTRA_LINKER_FLAGS_MSVC}  /SAFESEH:NO ")
    set( ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${EXTRA_LIBRARIES_MSVC} )
endif( )

set(POS_TESTS TestRungeKutta4thOrder 
		TestAdamsMoulton 
		TestParODEInterface)

foreach( project_name ${POS_TESTS})
	add_executable(${project_name} ${POS_SOURCE_FILES} 
					${POS_DEBUG_FILES} ${POS_TESTSSOURCE_FOLDER}/${project_name}.cpp)
	set_target_properties( ${project_name} PROPERTIES
                        			COMPILE_FLAGS ${COMPILER_FLAGS}
                        			LINK_FLAGS ${LINKER_FLAGS} )
	target_link_libraries( ${project_name} ${OPENCL_LIBRARIES} ${ADDITIONAL_LIBRARIES} )
endforeach( project_name )

# Set output directory to bin
if( MSVC )
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${BITNESS_SUFFIX})
else()
	set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/${BITNESS_SUFFIX}/${CMAKE_BUILD_TYPE})
endif()

# Copy kernel files to binary directory
foreach( extra_file ${POS_KERNEL_FILES} )
	foreach( project_name ${POS_TESTS})
    		add_custom_command(
        		TARGET ${project_name} POST_BUILD
        		COMMAND ${CMAKE_COMMAND} -E copy_if_different
        		${extra_file}  ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
        		${extra_file}  ./
        	)
	endforeach( project_name )
endforeach( extra_file )

# Create the shared library
add_library(ParODEShared SHARED ${POS_SOURCE_FILES} 
					${POS_DEBUG_FILES})
target_link_libraries(ParODEShared ${OPENCL_LIBRARIES} ${ADDITIONAL_LIBRARIES})

add_executable(TestParODESharedLibInterface ${POS_TESTSSOURCE_FOLDER}/TestParODEStaticLibCInterface.cpp)
set_target_properties(TestParODESharedLibInterface PROPERTIES 
						COMPILE_FLAGS ${COMPILER_FLAGS}
						LINK_FLAGS ${LINKER_FLAGS} )
target_link_libraries(TestParODESharedLibInterface ParODEShared)

