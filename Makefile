all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev -G Ninja && cmake --build build --parallel

debug:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -Wno-dev -G Ninja && cmake --build build --parallel

clean:
	rm -rf build
