all:
	mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -Wno-dev && ${MAKE}

clean:
	if [ -d "build" ]; then ${MAKE} clean -C build; fi
	rm -rf build
