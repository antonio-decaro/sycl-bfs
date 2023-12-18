link_libraries(sygraph)

include_directories(include)
add_executable(main src/main.cpp)
get_directory_property(all_targets BUILDSYSTEM_TARGETS)