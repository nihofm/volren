all:
	ninja -C build

release:
	cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-dev

debug:
	cmake -S . -B build -G Ninja  -DCMAKE_BUILD_TYPE=Debug -Wno-dev

clean:
	rm -rf build
