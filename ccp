In file included from new.c:1:
/usr/include/hip/amd_detail/amd_hip_runtime.h:64:1: error: unknown type name ‘size_t’
   64 | size_t amd_dbgapi_get_build_id();
      | ^~~~~~
/usr/include/hip/amd_detail/amd_hip_runtime.h:33:1: note: ‘size_t’ is defined in header ‘<stddef.h>’; did you forget to ‘#include <stddef.h>’?
   32 | #include <hip/amd_detail/amd_hip_common.h>
  +++ |+#include <stddef.h>
   33 | 
