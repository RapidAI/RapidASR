# OpenFST Windows Port Bugs And Limitations

"Fixed in" means that all previous releases are likely affected.

## Fixed in win/1.7.2.1+01

* stdout newline conversion of binary data (#20)

## Unimplemented features

* Memory-mapped files are not supported (we may add the support in the future
  though), because it is very system-dependent. OpenFST supports reading e. g.
  CompactFST files into allocated memory when memory mapping is not compiled in.
  Since Kaldi is now using it, this is on an implementation track, issue #31.

* Dynamic registration of arc and FST types is not supported in the Visual Studio
  project versions (as they build only static libraries). CMake build does not
  have this limitation. Due to ABI being specific to Microsoft compiler version,
  dynamically registered types must be compiled with strictly the same compiler
  of the same major version, and mostly same build flags. This is quite hard to
  get right, and is not recommended. No current plans to implement.
