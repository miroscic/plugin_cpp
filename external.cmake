include(ExternalProject)

ExternalProject_Add(armadillo
  PREFIX ${EXTERNAL_DIR}
  SOURCE_DIR ${EXTERNAL_DIR}/armadillo
  INSTALL_DIR ${USR_DIR}
  URL https://sourceforge.net/projects/arma/files/armadillo-12.6.7.tar.xz
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SMOKE_TEST=OFF
    -DCMAKE_CXX_STANDARD=17
    $<$<BOOL:${WIN32}>:-Dopenblas_LIBRARY=${EXTERNAL_DIR}/armadillo/examples/lib_win64/libopenblas.lib>
    -DOPENBLAS_PROVIDES_LAPACK=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${USR_DIR}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Install custom built OpenBLAS provided by armadillo
if(WIN32)
  file(INSTALL ${EXTERNAL_DIR}/armadillo/examples/lib_win64/libopenblas.dll 
    DESTINATION ${USR_DIR}/bin
  )
endif()

ExternalProject_Add(ensmallen
  PREFIX ${EXTERNAL_DIR}
  SOURCE_DIR ${EXTERNAL_DIR}/ensmallen
  INSTALL_DIR ${USR_DIR}
  GIT_REPOSITORY https://github.com/mlpack/ensmallen.git
  GIT_TAG 2.21.0
  CMAKE_ARGS 
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_INSTALL_PREFIX:PATH=${USR_DIR}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
add_dependencies(ensmallen armadillo)

ExternalProject_Add(cereal
  PREFIX ${EXTERNAL_DIR}
  SOURCE_DIR ${EXTERNAL_DIR}/cereal
  INSTALL_DIR ${USR_DIR}
  GIT_REPOSITORY https://github.com/USCiLab/cereal.git
  GIT_TAG v1.3.2
  CMAKE_ARGS 
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_INSTALL_PREFIX:PATH=${USR_DIR}
    -DBUILD_DOC=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_SANDBOX=OFF
    -DSKIP_PERFORMANCE_COMPARISON=ON
    -DWITH_WERROR=OFF
    -DCLANG_USE_LIBCPP=ON
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

ExternalProject_Add(mlpack
  PREFIX ${EXTERNAL_DIR}
  SOURCE_DIR ${EXTERNAL_DIR}/mlpack
  INSTALL_DIR ${USR_DIR}
  GIT_REPOSITORY https://github.com/mlpack/mlpack.git
  GIT_TAG 4.3.0
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_INSTALL_PREFIX:PATH=${USR_DIR}
    -DBUILD_CLI_EXECUTABLES=OFF
    -DBUILD_TESTS=OFF
    -DENSMALLEN_INCLUDE_DIR=${USR_DIR}/include
    -DCEREAL_INCLUDE_DIR=${USR_DIR}/include
    -DDOWNLOAD_DEPENDENCIES=ON
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
add_dependencies(mlpack ensmallen cereal)