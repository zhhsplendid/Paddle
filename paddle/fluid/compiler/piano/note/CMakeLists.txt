cc_library(note_opcode SRCS opcode.cc DEPS errors enforce)
cc_test(note_opcode_test SRCS opcode_test.cc DEPS note_opcode)

proto_library(note_proto SRCS note.proto)
set_target_properties(note_proto PROPERTIES COMPILE_FLAGS "-Wno-extra")
