add_executable(test_shader_types shader_types_test.cpp)
NoGraphicsAPI_disable_exceptions(test_shader_types)
target_link_libraries(test_shader_types PRIVATE NoGraphicsAPIUtility::types)
add_test(NAME test_shader_types COMMAND test_shader_types)

if(NOT COMMAND NoGraphicsAPI_compile_slang)
    include(${PROJECT_SOURCE_DIR}/examples/NoGraphicsAPI_slang.cmake)
endif()
# The raw Vulkan harness exercises these kernels independently of the native backend.
if(NOT APPLE)
    set(raytrace_source ${PROJECT_SOURCE_DIR}/examples/raytrace)
    set(raytrace_shader_dir ${CMAKE_CURRENT_BINARY_DIR}/shaders/raytrace)
    set(raytrace_spirv)
    set(raytrace_common ${raytrace_source}/raytrace_types.h ${raytrace_source}/primitive.slang ${raytrace_source}/transform.slang)
    set(raytrace_query ${raytrace_common} ${raytrace_source}/bvh.slang ${raytrace_source}/query.slang
        ${raytrace_source}/ray_functions.slang ${raytrace_source}/ray_functions_shared.h)
    foreach(shader IN ITEMS morton sort ploc refit import refit_import batch export)
        set(output ${raytrace_shader_dir}/${shader}.comp.spv)
        NoGraphicsAPI_compile_slang(${output} ${raytrace_source}/${shader}.slang computeMain compute
            DEFINES NOGRAPHICSAPI_ROOT_PHYSICAL DEPENDS ${raytrace_source}/${shader}_shared.h ${raytrace_source}/ploc_shared.h
                ${raytrace_source}/ploc_common.slang ${raytrace_source}/morton_code.slang ${raytrace_source}/bounds_encoding.slang ${raytrace_common})
        list(APPEND raytrace_spirv ${output})
    endforeach()
    NoGraphicsAPI_compile_slang(${raytrace_shader_dir}/query.comp.spv ${CMAKE_CURRENT_SOURCE_DIR}/raytrace_query.slang computeMain compute
        DEFINES NOGRAPHICSAPI_ROOT_PHYSICAL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/raytrace_query_shared.h ${raytrace_query})
    add_executable(test_raytrace raytrace_test.cpp raytrace_feature_cases.hpp raytrace_import_cases.hpp
        raytrace_sort_commands.hpp raytrace_sort_cases.hpp raytrace_batch_cases.hpp raytrace_import_batch_cases.hpp ${raytrace_source}/build.cpp
        ${raytrace_spirv} ${raytrace_shader_dir}/query.comp.spv)
    NoGraphicsAPI_disable_exceptions(test_raytrace)
    target_include_directories(test_raytrace PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
    target_link_libraries(test_raytrace PRIVATE NoGraphicsAPIUtility::types NoGraphicsAPIUtility::allocators Vulkan::Vulkan)
    if(WIN32)
        target_compile_definitions(test_raytrace PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME test_raytrace COMMAND test_raytrace ${raytrace_shader_dir})
    set_tests_properties(test_raytrace PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 FAIL_REGULAR_EXPRESSION "Raytrace validation:")
endif()

add_executable(test_math math_test.cpp)
NoGraphicsAPI_disable_exceptions(test_math)
target_link_libraries(test_math PRIVATE NoGraphicsAPIUtility::math)
add_test(NAME test_math COMMAND test_math)

add_executable(test_allocators ../utility/tests/allocator_test.cpp)
NoGraphicsAPI_disable_exceptions(test_allocators)
find_package(Threads REQUIRED)
target_link_libraries(test_allocators PRIVATE NoGraphicsAPIUtility::allocators Threads::Threads)
add_test(NAME test_allocators COMMAND test_allocators)

add_executable(test_upload_queue ../utility/tests/upload_queue_test.cpp)
NoGraphicsAPI_disable_exceptions(test_upload_queue)
target_link_libraries(test_upload_queue PRIVATE NoGraphicsAPIUtility::uploads)
if(NOT COMMAND NoGraphicsAPI_compile_slang)
    include(${PROJECT_SOURCE_DIR}/examples/NoGraphicsAPI_slang.cmake)
endif()
set(upload_queue_shader ${CMAKE_CURRENT_SOURCE_DIR}/shaders/upload_queue.slang)
set(upload_queue_spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/upload_queue.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
NoGraphicsAPI_compile_slang(${upload_queue_spv} ${upload_queue_shader} computeMain compute
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/upload_queue_shared.h)
target_sources(test_upload_queue PRIVATE ${upload_queue_spv})
target_compile_definitions(test_upload_queue PRIVATE NOGRAPHICSAPI_UPLOAD_QUEUE_SPV_PATH="${upload_queue_spv}")
add_test(NAME test_upload_queue COMMAND test_upload_queue)
add_test(NAME test_upload_queue_families COMMAND test_upload_queue --queue-families)
set_tests_properties(test_upload_queue test_upload_queue_families PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "NoGraphicsAPI validation:")

add_executable(test_texture_upload_tiles texture_upload_tiles_test.cpp)
NoGraphicsAPI_disable_exceptions(test_texture_upload_tiles)
target_link_libraries(test_texture_upload_tiles PRIVATE NoGraphicsAPI::NoGraphicsAPI)
add_test(NAME test_texture_upload_tiles COMMAND test_texture_upload_tiles)

if(WIN32)
    add_executable(test_math_win32_include math_win32_include_test.cpp)
    NoGraphicsAPI_disable_exceptions(test_math_win32_include)
    target_link_libraries(test_math_win32_include PRIVATE NoGraphicsAPIUtility::math)
    add_test(NAME test_math_win32_include COMMAND test_math_win32_include)
endif()

add_executable(test_api api_test.cpp)
NoGraphicsAPI_disable_exceptions(test_api)
target_link_libraries(test_api PRIVATE NoGraphicsAPI::NoGraphicsAPI)
add_test(NAME test_api COMMAND test_api)

add_executable(test_command_context command_context_test.cpp)
NoGraphicsAPI_disable_exceptions(test_command_context)
target_link_libraries(test_command_context PRIVATE NoGraphicsAPI::NoGraphicsAPI)
add_test(NAME test_command_context COMMAND test_command_context)

add_executable(test_placed_texture placed_texture_test.cpp)
NoGraphicsAPI_disable_exceptions(test_placed_texture)
target_link_libraries(test_placed_texture PRIVATE NoGraphicsAPIUtility::textures)
add_test(NAME test_placed_texture COMMAND test_placed_texture)

add_executable(test_multithreading multithreading_test.cpp)
NoGraphicsAPI_disable_exceptions(test_multithreading)
target_link_libraries(test_multithreading PRIVATE NoGraphicsAPI::NoGraphicsAPI Threads::Threads)

if(NOT APPLE)
    # Manual device-loss reproducer: no library implementation, shaders, or timestamp queries.
    add_executable(repro_queue_device_lost EXCLUDE_FROM_ALL repro_queue_device_lost.cpp)
    set_target_properties(repro_queue_device_lost PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/repro")
    NoGraphicsAPI_disable_exceptions(repro_queue_device_lost)
    target_compile_features(repro_queue_device_lost PRIVATE cxx_std_20)
    target_include_directories(repro_queue_device_lost PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(repro_queue_device_lost PRIVATE Vulkan::Vulkan Threads::Threads)
else()
    # Manual native timestamp diagnostic, excluded from normal builds and CTest.
    add_executable(repro_metal_render_timestamps EXCLUDE_FROM_ALL repro_metal_render_timestamps.mm)
    set_target_properties(repro_metal_render_timestamps PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/repro")
    NoGraphicsAPI_disable_exceptions(repro_metal_render_timestamps)
    target_compile_features(repro_metal_render_timestamps PRIVATE cxx_std_20)
    target_compile_options(repro_metal_render_timestamps PRIVATE -fno-objc-arc)
    target_include_directories(repro_metal_render_timestamps PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(repro_metal_render_timestamps PRIVATE "-framework Foundation" "-framework Metal")
endif()

add_test(NAME test_multithreading COMMAND test_multithreading)
add_test(NAME test_multiple_queues COMMAND test_multithreading --multiple-queues)
add_test(NAME test_queue_family_threads COMMAND test_multithreading --queue-families)

set_tests_properties(test_command_context test_placed_texture test_multithreading test_multiple_queues test_queue_family_threads PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

set(queue_family_spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/queue_family.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
NoGraphicsAPI_compile_slang(${queue_family_spv} ${CMAKE_CURRENT_SOURCE_DIR}/queue_family.slang computeMain compute
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/queue_family_shared.h)
add_executable(test_queue_families queue_family_test.cpp ${queue_family_spv})
NoGraphicsAPI_disable_exceptions(test_queue_families)
target_link_libraries(test_queue_families PRIVATE NoGraphicsAPI::NoGraphicsAPI NoGraphicsAPIUtility::types)
target_compile_definitions(test_queue_families PRIVATE NOGRAPHICSAPI_QUEUE_FAMILY_SPV="${queue_family_spv}")
add_test(NAME test_queue_families COMMAND test_queue_families)
add_test(NAME test_compute_queue COMMAND test_queue_families --compute)
add_test(NAME test_copy_queue COMMAND test_queue_families --copy)
set_tests_properties(test_queue_families test_compute_queue test_copy_queue PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

set(continuation_vertex_spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/render_continuation.vert.${NOGRAPHICSAPI_SHADER_EXTENSION})
set(continuation_fragment_spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/render_continuation.frag.${NOGRAPHICSAPI_SHADER_EXTENSION})
NoGraphicsAPI_compile_slang(${continuation_vertex_spv} ${CMAKE_CURRENT_SOURCE_DIR}/render_continuation.slang vertexMain vertex)
NoGraphicsAPI_compile_slang(${continuation_fragment_spv} ${CMAKE_CURRENT_SOURCE_DIR}/render_continuation.slang fragmentMain fragment)
add_executable(test_render_continuation render_continuation_test.cpp ${continuation_vertex_spv} ${continuation_fragment_spv})
NoGraphicsAPI_disable_exceptions(test_render_continuation)
target_link_libraries(test_render_continuation PRIVATE NoGraphicsAPI::NoGraphicsAPI Threads::Threads)
target_compile_definitions(test_render_continuation PRIVATE
    NOGRAPHICSAPI_CONTINUATION_VERTEX_SPV="${continuation_vertex_spv}"
    NOGRAPHICSAPI_CONTINUATION_FRAGMENT_SPV="${continuation_fragment_spv}")
add_test(NAME test_render_continuation COMMAND test_render_continuation)
set_tests_properties(test_render_continuation PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

if(WIN32)
    add_executable(test_presentation presentation_test.cpp)
    NoGraphicsAPI_disable_exceptions(test_presentation)
    target_link_libraries(test_presentation PRIVATE NoGraphicsAPI::NoGraphicsAPI)
    add_test(NAME test_presentation COMMAND test_presentation)
    add_test(NAME test_presentation_queue_families COMMAND test_presentation --queue-families)
    set_tests_properties(test_presentation test_presentation_queue_families PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "NoGraphicsAPI validation:")
endif()

add_executable(test_texture_copy texture_copy_test.cpp)
NoGraphicsAPI_disable_exceptions(test_texture_copy)
target_link_libraries(test_texture_copy PRIVATE NoGraphicsAPI::NoGraphicsAPI)
add_test(NAME test_texture_copy COMMAND test_texture_copy)
set_tests_properties(test_texture_copy PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_executable(test_metal_resource_concurrency metal_resource_concurrency_test.cpp ${queue_family_spv})
    NoGraphicsAPI_disable_exceptions(test_metal_resource_concurrency)
    target_link_libraries(test_metal_resource_concurrency PRIVATE NoGraphicsAPI::NoGraphicsAPI NoGraphicsAPIUtility::types Threads::Threads)
    target_compile_definitions(test_metal_resource_concurrency PRIVATE NOGRAPHICSAPI_CONCURRENCY_SHADER="${queue_family_spv}")
    add_test(NAME test_metal_address_registry COMMAND test_metal_resource_concurrency registry)
    add_test(NAME test_metal_timestamp_concurrency COMMAND test_metal_resource_concurrency timestamps)
    add_test(NAME test_metal_descriptor_concurrency COMMAND test_metal_resource_concurrency descriptors)
    set_tests_properties(test_metal_address_registry test_metal_timestamp_concurrency test_metal_descriptor_concurrency PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 180 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:"
        ENVIRONMENT "MTL_DEBUG_LAYER=1")
endif()

if(APPLE)
    function(NoGraphicsAPI_compile_metal_fixture output source)
        get_filename_component(output_dir "${output}" DIRECTORY)
        add_custom_command(
            OUTPUT ${output}
            BYPRODUCTS ${output}.air
            COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
            COMMAND ${NOGRAPHICSAPI_XCRUN} -sdk macosx metal -std=metal4.0 -c ${source} -o ${output}.air
            COMMAND ${NOGRAPHICSAPI_XCRUN} -sdk macosx metallib ${output}.air -o ${output}
            DEPENDS ${source} ${ARGN}
            VERBATIM
            COMMENT "Compiling native Metal test fixture ${source}"
        )
    endfunction()

    set(metal_heap_shader ${CMAKE_CURRENT_BINARY_DIR}/shaders/metal_heap.metallib)
    set(metal_draw_shader ${CMAKE_CURRENT_BINARY_DIR}/shaders/metal_draw.metallib)
    set(metal_mesh_shader ${CMAKE_CURRENT_BINARY_DIR}/shaders/metal_mesh.metallib)
    NoGraphicsAPI_compile_metal_fixture(${metal_heap_shader} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_heap.metal)
    NoGraphicsAPI_compile_metal_fixture(${metal_draw_shader} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_draw.metal)
    NoGraphicsAPI_compile_metal_fixture(${metal_mesh_shader} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_mesh.metal)
    set(metal_barrier_compute ${CMAKE_CURRENT_BINARY_DIR}/shaders/metal_barrier_compute.metallib)
    set(metal_barrier_graphics ${CMAKE_CURRENT_BINARY_DIR}/shaders/metal_barrier_graphics.metallib)
    NoGraphicsAPI_compile_metal_fixture(${metal_barrier_compute} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_barrier_compute.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_barrier_shared.h)
    NoGraphicsAPI_compile_metal_fixture(${metal_barrier_graphics} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_barrier_graphics.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal_barrier_shared.h)
    add_executable(test_metal_barriers metal_barrier_test.cpp ${metal_barrier_compute} ${metal_barrier_graphics})
    NoGraphicsAPI_disable_exceptions(test_metal_barriers)
    target_link_libraries(test_metal_barriers PRIVATE NoGraphicsAPI::NoGraphicsAPI)
    target_compile_definitions(test_metal_barriers PRIVATE
        NOGRAPHICSAPI_METAL_BARRIER_COMPUTE="${metal_barrier_compute}" NOGRAPHICSAPI_METAL_BARRIER_GRAPHICS="${metal_barrier_graphics}")
    add_test(NAME test_metal_barriers COMMAND test_metal_barriers)
    add_test(NAME test_metal_barriers_timestamps COMMAND test_metal_barriers --timestamps)
    set_tests_properties(test_metal_barriers test_metal_barriers_timestamps PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

    add_test(NAME test_metal_autorelease COMMAND test_command_context)
    set_tests_properties(test_metal_autorelease PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "MISSING POOLS:"
        ENVIRONMENT "OBJC_DEBUG_MISSING_POOLS=YES;MTL_DEBUG_LAYER=0;MTL_SHADER_VALIDATION=0")

    add_executable(test_metal_presentation metal_presentation_test.mm)
    NoGraphicsAPI_disable_exceptions(test_metal_presentation)
    target_compile_options(test_metal_presentation PRIVATE -fno-objc-arc)
    target_link_libraries(test_metal_presentation PRIVATE NoGraphicsAPI::NoGraphicsAPI)
    add_test(NAME test_metal_presentation COMMAND test_metal_presentation)
    set_tests_properties(test_metal_presentation PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

    add_executable(test_metal_backend metal_backend_test.cpp ${metal_heap_shader} ${metal_draw_shader})
    NoGraphicsAPI_disable_exceptions(test_metal_backend)
    target_link_libraries(test_metal_backend PRIVATE NoGraphicsAPI::NoGraphicsAPI)
    target_compile_definitions(test_metal_backend PRIVATE
        NOGRAPHICSAPI_METAL_HEAP_SHADER="${metal_heap_shader}" NOGRAPHICSAPI_METAL_DRAW_SHADER="${metal_draw_shader}")
    add_test(NAME test_metal_backend COMMAND test_metal_backend)
    set_tests_properties(test_metal_backend PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")
    add_executable(test_metal_mesh metal_mesh_test.cpp ${metal_mesh_shader})
    NoGraphicsAPI_disable_exceptions(test_metal_mesh)
    target_link_libraries(test_metal_mesh PRIVATE NoGraphicsAPI::NoGraphicsAPI)
    target_compile_definitions(test_metal_mesh PRIVATE NOGRAPHICSAPI_METAL_MESH_SHADER="${metal_mesh_shader}")
    add_test(NAME test_metal_mesh COMMAND test_metal_mesh)
    # Metal's GPU validator crashes on indirect mesh draws. Keep full coverage and validate direct draws separately.
    set_tests_properties(test_metal_mesh PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:"
        ENVIRONMENT "MTL_SHADER_VALIDATION=0")
    add_test(NAME test_metal_mesh_validation COMMAND test_metal_mesh --direct-only)
    set_tests_properties(test_metal_mesh_validation PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:"
        ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1")
endif()

if(NOT COMMAND NoGraphicsAPI_compile_slang)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../examples/NoGraphicsAPI_slang.cmake)
endif()
set(task_shader_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/task_shader.task.${NOGRAPHICSAPI_SHADER_EXTENSION})
set(task_mesh_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/task_mesh.mesh.${NOGRAPHICSAPI_SHADER_EXTENSION})
set(task_fragment_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/task_mesh.frag.${NOGRAPHICSAPI_SHADER_EXTENSION})
NoGraphicsAPI_compile_slang(${task_shader_code} ${CMAKE_CURRENT_SOURCE_DIR}/task_shader.slang taskMain amplification
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/task_shader_shared.h ${CMAKE_CURRENT_SOURCE_DIR}/task_shader_common.h)
NoGraphicsAPI_compile_slang(${task_mesh_code} ${CMAKE_CURRENT_SOURCE_DIR}/task_mesh.slang meshMain mesh
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/task_mesh_shared.h ${CMAKE_CURRENT_SOURCE_DIR}/task_shader_common.h)
NoGraphicsAPI_compile_slang(${task_fragment_code} ${CMAKE_CURRENT_SOURCE_DIR}/task_mesh.slang fragmentMain fragment
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/task_mesh_shared.h ${CMAKE_CURRENT_SOURCE_DIR}/task_shader_common.h)
add_executable(test_task_shader task_shader_test.cpp ${task_shader_code} ${task_mesh_code} ${task_fragment_code})
NoGraphicsAPI_disable_exceptions(test_task_shader)
target_link_libraries(test_task_shader PRIVATE NoGraphicsAPI::NoGraphicsAPI NoGraphicsAPIUtility::types)
target_compile_definitions(test_task_shader PRIVATE
    NOGRAPHICSAPI_TASK_TEST_TASK_SHADER="${task_shader_code}"
    NOGRAPHICSAPI_TASK_TEST_MESH_SHADER="${task_mesh_code}"
    NOGRAPHICSAPI_TASK_TEST_FRAGMENT_SHADER="${task_fragment_code}")
add_test(NAME test_task_shader COMMAND test_task_shader)
set_tests_properties(test_task_shader PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

if(APPLE)
    # The same MetalTools indirect-mesh validator failure affects Slang-generated task/mesh pipelines.
    set_tests_properties(test_task_shader PROPERTIES ENVIRONMENT "MTL_SHADER_VALIDATION=0")
    add_test(NAME test_task_shader_gpu_validation COMMAND test_task_shader --direct-only)
    set_tests_properties(test_task_shader_gpu_validation PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:"
        ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1")
endif()

set(root_data_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/root_data.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
set(root_aligned_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/root_data_aligned.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
set(root_generate_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/root_data_generate.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
NoGraphicsAPI_compile_slang(${root_data_code} ${CMAKE_CURRENT_SOURCE_DIR}/root_data.slang computeMain compute
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/root_data_shared.h)
NoGraphicsAPI_compile_slang(${root_aligned_code} ${CMAKE_CURRENT_SOURCE_DIR}/root_data.slang computeMain compute
    DEFINES ROOT_DATA_ALIGNMENT=16 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/root_data_shared.h)
NoGraphicsAPI_compile_slang(${root_generate_code} ${CMAKE_CURRENT_SOURCE_DIR}/root_data_generate.slang computeMain compute
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/root_data_generate_shared.h ${CMAKE_CURRENT_SOURCE_DIR}/root_data_shared.h)
add_executable(test_root_data root_data_test.cpp ${root_data_code} ${root_aligned_code} ${root_generate_code})
NoGraphicsAPI_disable_exceptions(test_root_data)
target_link_libraries(test_root_data PRIVATE NoGraphicsAPI::NoGraphicsAPI NoGraphicsAPIUtility::allocators)
target_compile_definitions(test_root_data PRIVATE NOGRAPHICSAPI_ROOT_DATA_PATH="${root_data_code}"
    NOGRAPHICSAPI_ROOT_GENERATE_PATH="${root_generate_code}")
add_test(NAME test_root_data COMMAND test_root_data)
add_test(NAME test_root_data_aligned COMMAND test_root_data ${root_aligned_code})
set_tests_properties(test_root_data test_root_data_aligned PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")

if(NOT APPLE)
    set(root_benchmark_shaders)
    foreach(mode IN ITEMS uniform structured physical push)
        string(TOUPPER ${mode} mode_define)
        foreach(bytes IN ITEMS 64 256)
            set(entry computeMain)
            if(bytes EQUAL 64)
                set(entry smallMain)
            endif()
            set(output ${CMAKE_CURRENT_BINARY_DIR}/root-benchmark/${mode}-${bytes}.spv)
            NoGraphicsAPI_compile_slang(${output} ${CMAKE_CURRENT_SOURCE_DIR}/root_data.slang ${entry} compute
                DEFINES ROOT_DATA_${mode_define} OPTIONS -O3 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/root_data_shared.h)
            list(APPEND root_benchmark_shaders ${output})
        endforeach()
    endforeach()
    foreach(mode IN ITEMS ssbo physical)
        string(TOUPPER ${mode} mode_define)
        foreach(alignment IN ITEMS 4 16)
            foreach(bytes IN ITEMS 64 256)
                set(entry computeMain)
                if(bytes EQUAL 64)
                    set(entry smallMain)
                endif()
                set(output ${CMAKE_CURRENT_BINARY_DIR}/root-benchmark/${mode}_aligned${alignment}-${bytes}.spv)
                NoGraphicsAPI_compile_slang(${output} ${CMAKE_CURRENT_SOURCE_DIR}/root_data.slang ${entry} compute
                    DEFINES NOGRAPHICSAPI_ROOT_${mode_define} ROOT_DATA_ALIGNMENT=${alignment}
                    OPTIONS -O3 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/root_data_shared.h)
                list(APPEND root_benchmark_shaders ${output})
            endforeach()
        endforeach()
    endforeach()
    add_executable(benchmark_root_data EXCLUDE_FROM_ALL root_data_benchmark.cpp ${root_benchmark_shaders})
    NoGraphicsAPI_disable_exceptions(benchmark_root_data)
    target_link_libraries(benchmark_root_data PRIVATE NoGraphicsAPI::NoGraphicsAPI NoGraphicsAPIUtility::allocators)
endif()

set(shader_abi_code ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader_abi.comp.${NOGRAPHICSAPI_SHADER_EXTENSION})
NoGraphicsAPI_compile_slang(${shader_abi_code} ${CMAKE_CURRENT_SOURCE_DIR}/shader_abi.slang computeMain compute
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shader_abi_shared.h)
add_executable(test_shader_abi shader_abi_test.cpp ${shader_abi_code})
NoGraphicsAPI_disable_exceptions(test_shader_abi)
target_link_libraries(test_shader_abi PRIVATE NoGraphicsAPI::NoGraphicsAPI NoGraphicsAPIUtility::types)
target_compile_definitions(test_shader_abi PRIVATE NOGRAPHICSAPI_SHADER_ABI_PATH="${shader_abi_code}")
add_test(NAME test_shader_abi COMMAND test_shader_abi)
set_tests_properties(test_shader_abi PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "Validation Error;NoGraphicsAPI validation:")
