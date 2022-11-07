set(LINK_LIBS diskpp)

add_executable(magnetostatics src/magnetostatics.cpp)
target_link_libraries(magnetostatics ${LINK_LIBS})
install(TARGETS magnetostatics RUNTIME DESTINATION bin)

install(DIRECTORY share/ DESTINATION share)