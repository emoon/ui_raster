#if defined(ISPC_TARGET_NEON)

int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    return return @llvm.aarch64.neon.sqrdmulh(a, diff, t);
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    return return @llvm.aarch64.neon.sqrdmulh(a, diff, t);
}

#else

int16 lerp_16x8_diff(int16 a, int16 diff, int16 t) {
    int16 temp = @llvm.x86.ssse3.pmul.hr.sw.128(diff, t);
    return a + temp;
}

int16 lerp_16x8(int16 a, int16 b, int16 t) {
    int16 diff = b - a;
    int16 temp = @llvm.x86.ssse3.pmul.hr.sw.128(diff, t);
    return a + temp;
}

#endif

export void ispc_raster() {
    print("Hello, from ispc!\n");
}
