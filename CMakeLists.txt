CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(RSFTest)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#include_directories("E:/packages/boost_1_47_0/boost_1_47_0/")
#link_directories("E:/packages/lib64/")

SET(Boost_USE_STATIC_LIBS   ON)
SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost COMPONENTS filesystem program_options)
FIND_PACKAGE(Boost REQUIRED COMPONENTS exception filesystem system program_options)
FIND_PACKAGE(Boost)
IF(Boost_FOUND)
 INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS})
  LINK_DIRECTORIES( ${Boost_LIBRARY_DIRS})
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR
          "Cannot build without boost. Please set BOOST_DIR.")
ENDIF(Boost_FOUND)

FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
          "Cannot build InsightApplications without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SET( CMAKE_CXX_FLAGS -DDLL_EXPORTS)		
IF(MSVC)
 SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
 SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
ENDIF(MSVC)								
											

#added
IF(UNIX)
  LINK_LIBRARIES( 
    boost_filesystem
    boost_program_options
  )
ENDIF(UNIX)
set(LevelSetsSourceList
  RSFTermTest
)
foreach( var ${LevelSetsSourceList} 
 )
  add_executable(${var} ${var}/${var}.cxx)
  target_link_libraries(${var} ${ITK_LIBRARIES} )
  endforeach()

