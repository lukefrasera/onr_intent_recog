project(intent_interface)

find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED COMPONENTS
  thread
  system
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
  include
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${CMAKE_SOURCE_DIR}/msg/intent.proto)



add_executable(simulate_csv_file
  src/csv_simulate.cc
  src/intent_interface.cc
  ${PROTO_SRC} ${PROTO_HEADER}
)
target_link_libraries(simulate_csv_file
  ${PROTOBUF_LIBRARIES}
)



add_executable(example_output
  src/examples/example_output.cc
)
target_link_libraries(example_output
  ${Boost_LIBRARIES}
)