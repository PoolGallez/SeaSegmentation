# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
include classes/FeatureExtractor/CMakeFiles/Extractor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include classes/FeatureExtractor/CMakeFiles/Extractor.dir/compiler_depend.make

# Include the progress variables for this target.
include classes/FeatureExtractor/CMakeFiles/Extractor.dir/progress.make

# Include the compile flags for this target's objects.
include classes/FeatureExtractor/CMakeFiles/Extractor.dir/flags.make

classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.o: classes/FeatureExtractor/CMakeFiles/Extractor.dir/flags.make
classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.o: ../classes/FeatureExtractor/Extractor.cpp
classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.o: classes/FeatureExtractor/CMakeFiles/Extractor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.o"
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.o -MF CMakeFiles/Extractor.dir/Extractor.cpp.o.d -o CMakeFiles/Extractor.dir/Extractor.cpp.o -c "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/classes/FeatureExtractor/Extractor.cpp"

classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Extractor.dir/Extractor.cpp.i"
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/classes/FeatureExtractor/Extractor.cpp" > CMakeFiles/Extractor.dir/Extractor.cpp.i

classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Extractor.dir/Extractor.cpp.s"
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/classes/FeatureExtractor/Extractor.cpp" -o CMakeFiles/Extractor.dir/Extractor.cpp.s

# Object files for target Extractor
Extractor_OBJECTS = \
"CMakeFiles/Extractor.dir/Extractor.cpp.o"

# External object files for target Extractor
Extractor_EXTERNAL_OBJECTS =

classes/FeatureExtractor/libExtractor.a: classes/FeatureExtractor/CMakeFiles/Extractor.dir/Extractor.cpp.o
classes/FeatureExtractor/libExtractor.a: classes/FeatureExtractor/CMakeFiles/Extractor.dir/build.make
classes/FeatureExtractor/libExtractor.a: classes/FeatureExtractor/CMakeFiles/Extractor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libExtractor.a"
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" && $(CMAKE_COMMAND) -P CMakeFiles/Extractor.dir/cmake_clean_target.cmake
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Extractor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
classes/FeatureExtractor/CMakeFiles/Extractor.dir/build: classes/FeatureExtractor/libExtractor.a
.PHONY : classes/FeatureExtractor/CMakeFiles/Extractor.dir/build

classes/FeatureExtractor/CMakeFiles/Extractor.dir/clean:
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" && $(CMAKE_COMMAND) -P CMakeFiles/Extractor.dir/cmake_clean.cmake
.PHONY : classes/FeatureExtractor/CMakeFiles/Extractor.dir/clean

classes/FeatureExtractor/CMakeFiles/Extractor.dir/depend:
	cd "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/classes/FeatureExtractor" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor" "/home/pool/Documents/Magistrale/Computer Vision/Boat Detection/Semantic Segmentation/BoW_Segmentation/build/classes/FeatureExtractor/CMakeFiles/Extractor.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : classes/FeatureExtractor/CMakeFiles/Extractor.dir/depend

