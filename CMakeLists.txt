cmake_minimum_required(VERSION 3.8)

set(BIN_INSTALL_DIR "./install")

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.9)
endif()

project(MyOptiXPrimeProject C CXX CUDA)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OptiX REQUIRED)

file(GLOB_RECURSE ${PROJECT_NAME}_sources "${PROJECT_SOURCE_DIR}/src/*.c*" "${PROJECT_SOURCE_DIR}/include/*.h*")

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_sources})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC ${OPTIX_INCLUDE_DIR})

if(WIN32)
	# Needed for CUDA apparently
	target_compile_definitions(${PROJECT_NAME} PUBLIC NOMINMAX _USE_MATH_DEFINES)
endif()

add_custom_target(CopyDLLs ALL COMMENT "Copying DLLs"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${OPTIX_DLL} $<TARGET_FILE_DIR:${PROJECT_NAME}>)

target_link_libraries(${PROJECT_NAME} ${OPTIX_LIB})

install(TARGETS ${PROJECT_NAME} DESTINATION ${BIN_INSTALL_DIR})