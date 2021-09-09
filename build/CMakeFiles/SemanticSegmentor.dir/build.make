# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build"

# Include any dependencies generated for this target.
include CMakeFiles/SemanticSegmentor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SemanticSegmentor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SemanticSegmentor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SemanticSegmentor.dir/flags.make

CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o: CMakeFiles/SemanticSegmentor.dir/flags.make
CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o: ../srcs/SemanticSegmentor.cpp
CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o: CMakeFiles/SemanticSegmentor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o -MF CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o.d -o CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o -c "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/srcs/SemanticSegmentor.cpp"

CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/srcs/SemanticSegmentor.cpp" > CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.i

CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/srcs/SemanticSegmentor.cpp" -o CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.s

# Object files for target SemanticSegmentor
SemanticSegmentor_OBJECTS = \
"CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o"

# External object files for target SemanticSegmentor
SemanticSegmentor_EXTERNAL_OBJECTS =

libSemanticSegmentor.a: CMakeFiles/SemanticSegmentor.dir/srcs/SemanticSegmentor.cpp.o
libSemanticSegmentor.a: CMakeFiles/SemanticSegmentor.dir/build.make
libSemanticSegmentor.a: CMakeFiles/SemanticSegmentor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libSemanticSegmentor.a"
	$(CMAKE_COMMAND) -P CMakeFiles/SemanticSegmentor.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SemanticSegmentor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SemanticSegmentor.dir/build: libSemanticSegmentor.a
.PHONY : CMakeFiles/SemanticSegmentor.dir/build

CMakeFiles/SemanticSegmentor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SemanticSegmentor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SemanticSegmentor.dir/clean

CMakeFiles/SemanticSegmentor.dir/depend:
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/CMakeFiles/SemanticSegmentor.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/SemanticSegmentor.dir/depend
