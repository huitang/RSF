CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(RSFTest)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

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

set(LevelSetsSourceList
  RSFTermTest
)
foreach( var ${LevelSetsSourceList}
 )
  add_executable(${var} ${var}/${var}.cxx)
  target_link_libraries(${var} ${ITK_LIBRARIES} ${Boost_LIBRARIES} )
endforeach()

include( CTest )
add_test( RSFTest ${CMAKE_BINARY_DIR}/RSFTermTest
  ${CMAKE_CURRENT_SOURCE_DIR}/TestImage/ini2D.mhd
  ${CMAKE_CURRENT_SOURCE_DIR}/TestImage/ori2D.mhd
  output.mhd 1 1 1 20 1000 2
)
