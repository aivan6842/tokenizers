CXX = g++
CXXFLAGS = -std=c++20 -Wall -Werror=vla -MMD -g -I ./src/
EXEC = main
OBJECTS = src/BPETokenizer.o src/main.o
DEPENDS = ${OBJECTS:.o=.d} 

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}
 
.PHONY: clean
 
clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}
