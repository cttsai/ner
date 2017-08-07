cmake_minimum_required(VERSION 3.7)
project(NER)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        Document.h
        main.cpp FeatureExtractor.cpp FeatureExtractor.h Utils.cpp Utils.h)

add_executable(NER ${SOURCE_FILES})
add_subdirectory(liblinear-poly2-2.01)

# for reading gazetteers
include_directories(/home/ctsai12/gzstream)
include_directories(/home/ctsai12/CLionProjects/NER/libs)


#add_library(linear liblinear-poly2-2.01/linear.cpp liblinear-poly2-2.01/tron.cpp)

#target_link_libraries(NER /home/ctsai12/CLionProjects/NER/libs/tron.o /home/ctsai12/CLionProjects/NER/libs/linear-poly2.o /home/ctsai12/CLionProjects/NER/libs/blas.a /home/ctsai12/gzstream/gzstream.o z)
target_link_libraries(NER /home/ctsai12/CLionProjects/NER/libs/tron.o /home/ctsai12/CLionProjects/NER/libs/linear-weight.o /home/ctsai12/CLionProjects/NER/libs/blas.a /home/ctsai12/gzstream/gzstream.o z)
#target_link_libraries(NER /home/ctsai12/CLionProjects/NER/libs/tron.o /home/ctsai12/CLionProjects/NER/libs/linear.o /home/ctsai12/CLionProjects/NER/libs/blas.a /home/ctsai12/gzstream/gzstream.o z)
#target_link_libraries(NER linear blas /home/ctsai12/gzstream/gzstream.o z)



