all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev && make -C build -j 17

debug:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -Wno-dev && make -C build -j 17

clean:
	rm -rf build
