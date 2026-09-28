set(shader_dir ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(types_header ${CMAKE_CURRENT_SOURCE_DIR}/raytrace_types.h)
set(raytrace_common ${types_header} ${CMAKE_CURRENT_SOURCE_DIR}/primitive.slang ${CMAKE_CURRENT_SOURCE_DIR}/transform.slang)
set(raytrace_query ${raytrace_common} ${CMAKE_CURRENT_SOURCE_DIR}/bvh.slang ${CMAKE_CURRENT_SOURCE_DIR}/query.slang
    ${CMAKE_CURRENT_SOURCE_DIR}/ray_functions.slang ${CMAKE_CURRENT_SOURCE_DIR}/ray_functions_shared.h)
set(morton_dependencies ${raytrace_common} ${CMAKE_CURRENT_SOURCE_DIR}/morton_code.slang)
set(sort_dependencies ${types_header})
set(ploc_dependencies ${raytrace_common} ${CMAKE_CURRENT_SOURCE_DIR}/ploc_common.slang)
set(refit_dependencies ${raytrace_common} ${CMAKE_CURRENT_SOURCE_DIR}/ploc_shared.h)
set(trace_dependencies ${raytrace_query})
set(import_dependencies ${types_header})
set(refit_import_dependencies ${raytrace_common} ${CMAKE_CURRENT_SOURCE_DIR}/bounds_encoding.slang)
set(export_dependencies ${types_header})
set(batch_dependencies ${raytrace_common} ${CMAKE_CURRENT_SOURCE_DIR}/morton_code.slang ${CMAKE_CURRENT_SOURCE_DIR}/bounds_encoding.slang)

set(kernel_sources)
set(kernel_definitions)
foreach(kernel IN ITEMS morton sort ploc refit trace import refit_import export batch)
    set(source ${CMAKE_CURRENT_SOURCE_DIR}/${kernel}.slang)
    set(shared ${CMAKE_CURRENT_SOURCE_DIR}/${kernel}_shared.h)
    set(output ${shader_dir}/${kernel}.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
    NoGraphicsAPI_compile_slang(${output} ${source} computeMain compute DEPENDS ${shared} ${${kernel}_dependencies})
    list(APPEND kernel_sources ${source} ${shared} ${output})
    string(TOUPPER ${kernel} name)
    list(APPEND kernel_definitions NOGRAPHICSAPI_${name}_SHADER_PATH="${output}")
endforeach()

NoGraphicsAPI_add_example(example_raytrace
    raytrace.cpp
    build.cpp
    build.hpp
    sort.hpp
    compute.hpp
    raytrace_types.h
    stb_image_write.h
    primitive.slang
    transform.slang
    query.slang
    ray_functions.slang
    ray_functions_shared.h
    bvh.slang
    ${kernel_sources}
)
target_link_libraries(example_raytrace PRIVATE NoGraphicsAPIUtility::allocators)
target_compile_definitions(example_raytrace PRIVATE ${kernel_definitions})
if(WIN32)
    target_compile_definitions(example_raytrace PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
