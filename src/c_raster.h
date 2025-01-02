#pragma once

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h> // For ARM NEON intrinsics

#define UIR_INLINE inline __attribute__((always_inline))

typedef int16x8_t uir_i16x8;
typedef float32x4_t uir_f32x4_t;
typedef int32x4_t uir_i32x4_t;

UIR_INLINE uir_i16x8 uir_i16x8_load_unaligned(const int16_t* data) {
    return vld1q_s16(data);
}

UIR_INLINE uir_i16x8 uir_i16x8_store_unaligned(const int16_t* data, uir_i16x8 value) {
    vst1q_s16((int16_t*)data, value);
    return value;
}

UIR_INLINE uir_i16x8 uir_i16x8_new_splat(int16_t value) {
    return vdupq_n_s16(value);
}

UIR_INLINE uir_i16x8 uir_i16x8_add(uir_i16x8 a, uir_i16x8 b) {
    return vaddq_s16(a, b);
}

UIR_INLINE uir_i16x8 uir_i16x8_sub(uir_i16x8 a, uir_i16x8 b) {
    return vsubq_s16(a, b);
}

UIR_INLINE uir_f32x4_t uir_i16x8_lerp(uir_i16x8 a, uir_i16x8 b, uir_i16x8 fraction) {
    uir_i16x8 diff = vsubq_s16(b, a);
    return vqrdmlahq_s16(a, diff, fraction); 
}

UIR_INLINE uir_f32x4_t uir_i32x4_load_unaligned(const int* data) {
    return vld1q_s32((int*)data);
}

UIR_INLINE uir_f32x4_t uir_f32x4_load_unaligned(const float* data) {
    return vld1q_f32((float*)data);
}

UIR_INLINE uir_f32x4_t uir_f32x4_new_splat(float value) {
    return vdupq_n_f32(value);
}

UIR_INLINE uir_f32x4_t uir_f32x4_sub(uir_f32x4_t a, uir_f32x4_t b) {
    return vsubq_f32(a, b);
}

UIR_INLINE uir_f32x4_t uir_f32x4_add(uir_f32x4_t a, uir_f32x4_t b) {
    return vaddq_f32(a, b);
}

UIR_INLINE uir_f32x4_t uir_f32x4_mul(uir_f32x4_t a, uir_f32x4_t b) {
    return vmulq_f32(a, b);
}

UIR_INLINE uir_f32x4_t uir_f32x4_floor(uir_f32x4_t a) {
    return vrndq_f32(a);
}

UIR_INLINE uir_i32x4_t uir_f32x4_to_i32x4(uir_f32x4_t a) {
    return vcvtq_s32_f32(a);
}

UIR_INLINE uir_f32x4_t uir_i32x4_to_f32x4(uir_i32x4_t a) {
    return vcvtq_s32_f32(a);
}

// macro because index has to be constant
#define uir_i32x4_extract(v, index) vgetq_lane_s32(v, index)

#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h> // For SSE4.2 intrinsics

#if defined(_MSC_VER) 
    #if defined(__clang__)
        #define UIR_INLINE inline __attribute__((always_inline))
    #else 
        #define UIR_INLINE __forceinline
    #endif
#else
    #define UIR_INLINE inline __attribute__((always_inline))
#endif

typedef __m128i uir_i16x8;
typedef __m128 uir_f32x4_t;
typedef __m128i uir_i32x4_t;

UIR_INLINE uir_i16x8 uir_i16x8_load_unaligned(const int16_t* data) {
    return _mm_loadu_si128((__m128i*)data);
}

UIR_INLINE uir_i16x8 uir_i16x8_store_unaligned(const int16_t* data, uir_i16x8 value) {
    _mm_storeu_si128((__m128i*)data, value);
    return value;
}

UIR_INLINE uir_i16x8 uir_i16x8_new_splat(int16_t value) { 
    return _mm_set1_epi16(value);
}

UIR_INLINE uir_i16x8 uir_i16x8_add(uir_i16x8 a, uir_i16x8 b) {
    return _mm_add_epi16(a, b);
}

UIR_INLINE uir_i16x8 uir_i16x8_sub(uir_i16x8 a, uir_i16x8 b) {
    return _mm_sub_epi16(a, b);
}

UIR_INLINE uir_i16x8 uir_i16x8_lerp(uir_i16x8 a, uir_i16x8 b, uir_i16x8 fraction) {
    return _mm_add_epi16(a, _mm_mulhrs_epi16(_mm_sub_epi16(b, a), fraction)); 
}

UIR_INLINE uir_f32x4_t uir_f32x4_load_unaligned(const float* data) {
    return _mm_loadu_ps(data);
}

UIR_INLINE uir_f32x4_t uir_f32x4_new_splat(float value) {
    return _mm_set1_ps(value);
}

UIR_INLINE uir_f32x4_t uir_f32x4_sub(uir_f32x4_t a, uir_f32x4_t b) {
    return _mm_sub_ps(a, b);
}

UIR_INLINE uir_f32x4_t uir_f32x4_add(uir_f32x4_t a, uir_f32x4_t b) {
    return _mm_add_ps(a, b);
}

UIR_INLINE uir_f32x4_t uir_f32x4_mul(uir_f32x4_t a, uir_f32x4_t b) {
    return _mm_mul_ps(a, b);
}

UIR_INLINE uir_f32x4_t uir_f32x4_floor(uir_f32x4_t a) {
    return _mm_floor_ps(a);
}

UIR_INLINE uir_i32x4_t uir_i32x4_load_unaligned(const int* data) {
    return _mm_loadu_si128((__m128i*)data);
}

UIR_INLINE uir_f32x4_t uir_i32x4_to_f32x4(uir_i32x4_t a) {
    return _mm_cvtepi32_ps(a);
}

UIR_INLINE uir_i32x4_t uir_f32x4_to_i32x4(uir_f32x4_t a) {
    return _mm_cvttps_epi32(a);
}

// macro because index has to be constant
#define uir_i32x4_extract(a, index) _mm_extract_epi32(a, index)

#else
#error "Unsupported architecture. Please compile for ARM or x86_64."
#endif

typedef struct uir_TileInfo {
    float offsets[4];
    int width;
    int height;
} uir_TileInfo;

typedef enum uir_PixelCount {
    PixelCount_Count1 = 1,
    PixelCount_Count2 = 2,
} uir_PixelCount;

UIR_INLINE uir_i16x8 process_pixel(
    uir_PixelCount pixel_count, 
    int16_t* output, 
    const int16_t* texture_data, 
    int texture_width,
    uir_i16x8 u_fraction,
    uir_i16x8 v_fraction)
{
    uir_i16x8 color;
    uir_i16x8 tex_rgba0;

    {
        uir_i16x8 tex_rgba_rgba_0 = uir_i16x8_load_unaligned(texture_data); 
        uir_i16x8 tex_rgba_rgba_1 = uir_i16x8_load_unaligned(texture_data + (texture_width * 4));
        uir_i16x8 t0_t1 = uir_i16x8_lerp(tex_rgba_rgba_0, tex_rgba_rgba_1, v_fraction);
        //__m128i t0 = __builtin_shufflevector(t0_t1, t0_t1, 0, 1, 2, 3, 0, 1, 2, 3);
        tex_rgba0 = uir_i16x8_lerp(t0_t1, t0_t1, u_fraction);
    }

    if (pixel_count == PixelCount_Count2)
    {
        uir_i16x8 tex_rgba_rgba_0 = uir_i16x8_load_unaligned(texture_data + 4); 
        uir_i16x8 tex_rgba_rgba_1 = uir_i16x8_load_unaligned(texture_data + (texture_width * 4) + 4);
        uir_i16x8 t0_t1 = uir_i16x8_lerp(tex_rgba_rgba_0, tex_rgba_rgba_1, v_fraction);
        //uir_i16x8 t0 = __builtin_shufflevector(t0_t1, t0_t1, 4, 5, 6, 7, 0, 1, 2, 3);
        uir_i16x8 tex_rgba1 = uir_i16x8_lerp(t0_t1, t0_t1, u_fraction);
        color = uir_i16x8_add(tex_rgba0, tex_rgba1) ;
    } else {
        color = tex_rgba0;
    }

    return color;
}

void internal_raster(
    int16_t* output, 
    const uir_TileInfo* tile_info,
    const int16_t* texture_data, 
    const int* texture_sizes_a,
    const float* coords,
    const float* uv_coords,
    const int16_t* top_colors_a,
    const int16_t* bottom_colors_a) 
{
    uir_i16x8 fixed_u_fraction;
    uir_i16x8 fixed_v_fraction;

    //uir_i16x8 top_colors = uir_i16x8_load_unaligned(top_colors_a);
    //uir_i16x8 bottom_colors = uir_i16x8_load_unaligned(bottom_colors_a);

    uir_f32x4_t x0y0x1y1_l = uir_f32x4_load_unaligned(coords);
    uir_f32x4_t tile_offsets = uir_f32x4_load_unaligned(tile_info->offsets);

    // Adjust for tile offset and center pixels at 0.5
    uir_f32x4_t x0y0x1y1_adjust = uir_f32x4_add(uir_f32x4_sub(x0y0x1y1_l, tile_offsets), uir_f32x4_new_splat(0.5f));
    uir_f32x4_t x0y0x1y1 = uir_f32x4_floor(x0y0x1y1_adjust);

    uir_i32x4_t texture_sizes = uir_i32x4_load_unaligned(texture_sizes_a);
    uir_f32x4_t texture_sizes_f = uir_i32x4_to_f32x4(texture_sizes); 

    // TODO: Bool check
    { 
        uir_f32x4_t uv0uv1 = uir_f32x4_load_unaligned(uv_coords); 
        uir_f32x4_t uv2uv3 = uir_f32x4_load_unaligned(uv_coords + 4);
        //
        uv0uv1 = uir_f32x4_mul(uv0uv1, texture_sizes_f);  
        uv2uv3 = uir_f32x4_mul(uv2uv3, texture_sizes_f);

        // uv_fraction *= 0x7fff
        uir_f32x4_t uv_fraction_f = uir_f32x4_mul(uir_f32x4_sub(x0y0x1y1_adjust, x0y0x1y1), uir_f32x4_new_splat(0x7fff));
        uir_i32x4_t uv_fraction_i = uir_f32x4_to_i32x4(uv_fraction_f);
        uir_i16x8 v_7fff = uir_i16x8_new_splat(0x7fff);
        
        fixed_u_fraction = uir_i16x8_sub(v_7fff, uv_fraction_i);
        fixed_v_fraction = uir_i16x8_sub(v_7fff, uv_fraction_i); 

        uir_i32x4_t uv0uv1_i = uir_f32x4_to_i32x4(uv0uv1);

        int texture_width = uir_i32x4_extract(texture_sizes, 0);
        int u0 = uir_i32x4_extract(uv0uv1_i, 0);
        int v0 = uir_i32x4_extract(uv0uv1_i, 1);
        int texture_start = v0 * texture_width + u0;
        texture_data += texture_start;
    }

    // TODO: Bool check
    {
    }

    // Get the coords to loop over
    uir_i32x4_t x0y0x1y1_i = uir_f32x4_to_i32x4(x0y0x1y1);
    const int x0 = uir_i32x4_extract(x0y0x1y1_i, 0);
    const int y0 = uir_i32x4_extract(x0y0x1y1_i, 1);
    const int x1 = uir_i32x4_extract(x0y0x1y1_i, 2);
    const int y1 = uir_i32x4_extract(x0y0x1y1_i, 3);

    const int xlen = x1 - x0;
    const int ylen = y1 - y0;
    int texture_width = uir_i32x4_extract(texture_sizes, 0);

    // Adjust starting position of the output data 
    output += y0 * tile_info->width + x0;

    for (int y = 0; y < ylen; ++y) {
        for (int x = 0; x < (xlen >> 1); x++) {
            uir_i16x8 color = process_pixel(PixelCount_Count2, output, texture_data, texture_width, fixed_u_fraction, fixed_v_fraction);
            uir_i16x8_store_unaligned(output, color);
            output += 8;
        }

        if (xlen & 1) {
            uir_i16x8 color = process_pixel(PixelCount_Count1, output, texture_data, texture_width, fixed_u_fraction, fixed_v_fraction);
            uir_i16x8_store_unaligned(output, color);
            output += 8;
        }

        output += (tile_info->width - xlen) * 8;
    }
}



