# Enable OpenMP
# If just -mp is specified, OMP_NUM_THREADS must be set in order to run in parallel
# Specifying 'allcores' will run on all cores if OMP_NUM_THREADS is not set (which seems
#  to be the default for other OpenMP implementations)
IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  IF(ENABLE_OFFLOAD AND NOT CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")
    IF(DEFINED OFFLOAD_ARCH)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu -gpu=${OFFLOAD_ARCH} -D__NO_UDR")
    ELSE()
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu -gpu=cc70 -D__NO_UDR")
    ENDIF()
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=allcores")
  ENDIF()
ENDIF(QMC_OMP)

ADD_DEFINITIONS( -Drestrict=__restrict__ )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__forceinline=inline")

# Set extra optimization specific flags
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fast" )


# Setting this to 'OFF' adds the -A flag, which enforces strict standard compliance
#  and causes the compilation to fail with some GNU header files
SET(CMAKE_CXX_EXTENSIONS ON)

# Add static flags if necessary
IF(QMC_BUILD_STATIC)
    SET(CMAKE_CXX_LINK_FLAGS " -Bstatic")
ENDIF(QMC_BUILD_STATIC)
