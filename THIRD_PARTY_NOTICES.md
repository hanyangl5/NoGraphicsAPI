# Third-party notices

## Vulkan-Tools cube texture

`examples/cube/lunarg_logo_256x256.rgba8` is derived from
`cube/lunarg.ppm.h` in the Vulkan-Tools vkcube sample at tag
`vulkan-sdk-1.4.357.0`:

https://github.com/KhronosGroup/Vulkan-Tools/blob/vulkan-sdk-1.4.357.0/cube/lunarg.ppm.h

The upstream texture carries these notices:

```text
Copyright (c) 2015-2019 The Khronos Group Inc.
Copyright (c) 2015-2019 Valve Corporation
Copyright (c) 2015-2019 LunarG, Inc.
```

The texture is redistributed under the Apache License, Version 2.0. A complete
copy is available in `examples/cube/LICENSE-Apache-2.0.txt`.

## stb_image_write

`examples/raytrace/stb_image_write.h` is stb_image_write v1.13 by Sean Barrett,
in the public domain:

http://nothings.org/stb/stb_image_write.h

## HIPRT

`examples/raytrace/ploc_common.slang` adapts locally ordered clustering from
[HIPRT's PlocBuilderKernels.h](https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/PlocBuilderKernels.h).
`examples/raytrace/transform.slang` follows the decomposition and interpolation conventions in HIPRT's
`Transform.h`, `QrDecomposition.h`, and `Quaternion.h`.

Copyright (C) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
The MIT license is included in `examples/raytrace/LICENSE-HIPRT.txt`.
