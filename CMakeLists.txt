CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(RSFTest)

FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
          "Cannot build RSFTest without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SET( CMAKE_CXX_FLAGS -DDLL_EXPORTS)

set(LevelSetsSourceList
  RSFTermTest
)
foreach( var ${LevelSetsSourceList}
 )
  add_executable(${var} ${var}/${var}.cxx)
  target_link_libraries(${var} ${ITK_LIBRARIES}  )
endforeach()

include( CTest )
add_test( RSFTest ${CMAKE_BINARY_DIR}/RSFTermTest
  ${RSFTest_SOURCE_DIR}/TestImage/ini2D.mhd
  ${RSFTest_SOURCE_DIR}/TestImage/ori2D.mhd
  output.mhd 1 1 1 20 1000 2
)
