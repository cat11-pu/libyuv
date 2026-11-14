# YUV conversion property test

`convert_property_test.cc` drives the row-based conversion functions changed
by the `Half` row conversion work. Each generated case uses:

* randomized width, height, vertical flip, and source/destination stride
  padding;
* randomized 8, 10, or 12 bit values over the function's full legal range;
* guard bands before and after every output allocation;
* sentinel bytes in every destination row's stride padding.

For each case and CPU mask the test checks:

1. no output guard or row-padding write;
2. two runs with differently initialized destination buffers are identical;
3. the optimized result is byte/pixel exact to the C fallback.

CPU masks are derived from the detected instruction set. On x86 this covers
the all-features path, AVX2, SSSE3, and the C/SSE2 fallback.

## Build and run

```sh
cmake -S . -B build -G Ninja -DPROPERTY_TEST=ON
cmake --build build --target convert_property_test
./build/convert_property_test --cases 10000 --seed 0x3712
```

Useful options:

```text
--cases N        number of randomized cases (default 10000)
--seed N         deterministic PRNG seed
--max-width N    maximum generated width
--max-height N   maximum generated height
--filter NAME    only run functions whose name contains NAME
--no-shrink      print the original failure without minimizing it
--inject-mismatch  harness self-test; verifies the shrink/report path
```

A failure is minimized over width, height, all four stride paddings, the
color-range limit, and the vertical-flip flag before it is printed.

## ASan variant

```sh
cmake -S . -B build-asan -G Ninja -DPROPERTY_TEST=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS='-fsanitize=address -fno-omit-frame-pointer' \
  -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer'
cmake --build build-asan --target convert_property_test
ASAN_OPTIONS=detect_leaks=0 ./build-asan/convert_property_test --cases 10000
```
