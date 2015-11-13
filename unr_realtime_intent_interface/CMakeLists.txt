project(intent_interface)

find_package(Protobuf REQUIRED)


include_directories(
  include
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${CMAKE_SOURCE_DIR}/msg/intent.proto)

add_executable(simulate_csv_file
  src/csv_simulate.cc
  ${PROTO_SRC} ${PROTO_HEADER}
)

target_link_libraries(simulate_csv_file
  ${PROTOBUF_LIBRARIES}
)