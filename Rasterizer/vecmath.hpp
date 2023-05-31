#ifndef VECMATH_HPP
#define VECMATH_HPP

#include <iterator>
#include <algorithm>
#include <cstring>
#include <array>
#include <cmath>
#include <concepts>

#define VM_PI 3.141592653f
#define VM_DEG_TO_RAD(deg) ((deg / 180.0f) * VM_PI)
#define VM_RAD_TO_DEG ((deg / VM_PI) * 180.0f)

#if defined(_MSC_VER)
#	include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#	include <x86intrin.h>
#endif

#include <xmmintrin.h>

namespace raw {
    template <typename T, size_t S>
    using vec = std::array<T, S>;

    template <typename T>
    concept ValidVector = requires(T t) {
        { std::size(t) } -> std::convertible_to<std::size_t>;
        { std::size(t) >= 2 && std::size(t) <= 4 } -> std::same_as<bool>;
    };

    template <typename T, size_t S>
    static vec<T, 4> pad(const vec<T, S>& orig) {
        vec<T, 4> ret;
        std::fill(ret.begin(), ret.end(), T(0));
        for (size_t i = 0; i < S; i++) ret[i] = orig[i];
        return ret;
    }

    template <typename T, size_t S>
    static T dot(const vec<T, S>& a, const vec<T, S>& b) requires ValidVector<vec<T, S>> {
        T ret{};
        for (size_t i = 0; i < S; i++) ret += a[i] * b[i];
        return ret;
    }

    template <>
    static float dot(const vec<float, 4>& a, const vec<float, 4>& b) {
        float ret;
        __m128 va = _mm_load_ps(a.data());
        __m128 vb = _mm_load_ps(b.data());
        _mm_store_ps(&ret, _mm_dp_ps(va, vb, 0xFF));
        return ret;
    }

    template <>
    static float dot(const vec<float, 3>& a, const vec<float, 3>& b) {
        float ret;
        __m128 va = _mm_load_ps(pad(a).data());
        __m128 vb = _mm_load_ps(pad(b).data());
        _mm_store_ps(&ret, _mm_dp_ps(va, vb, 0xFF));
        return ret;
    }

    template <typename T, size_t S>
    static T lengthSqr(const vec<T, S>& a) requires ValidVector<vec<T, S>> {
        return dot(a, a);
    }

    template <typename T, size_t S>
    static T length(const vec<T, S>& a) requires ValidVector<vec<T, S>> {
        return std::sqrt(lengthSqr(a));
    }

    template <typename T, size_t S>
    static vec<T, S> operator +(const vec<T, S>& a, const vec<T, S>& b) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = a[i] + b[i];
        return ret;
    }

    template <typename T, size_t S>
    static vec<T, S> operator -(const vec<T, S>& a, const vec<T, S>& b) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = a[i] - b[i];
        return ret;
    }

    template <typename T, size_t S>
    static vec<T, S> operator *(const vec<T, S>& a, const vec<T, S>& b) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = a[i] * b[i];
        return ret;
    }

    template <typename T, size_t S>
    static vec<T, S> operator *(const vec<T, S>& a, T b) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = a[i] * b;
        return ret;
    }

    template <typename T, size_t S>
    static vec<T, S> operator /(const vec<T, S>& a, T b) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = a[i] / b;
        return ret;
    }

    template <typename T, size_t S>
    static vec<T, S> operator /(T a, const vec<T, S>& b) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = a / b[i];
        return ret;
    }

    template <typename T, size_t S>
    static vec<T, S> normalized(const vec<T, S>& a) requires ValidVector<vec<T, S>> {
        return a / length(a);
    }

    template <typename T, size_t S>
    static vec<T, S> lerp(const vec<T, S>& a, const vec<T, S>& b, float t) requires ValidVector<vec<T, S>> {
        return a * (1.0f - t) + b * t;
    }

    template <typename T, size_t S>
    static vec<T, S> saturate(const vec<T, S>& a) requires ValidVector<vec<T, S>> {
        vec<T, S> ret{};
        for (size_t i = 0; i < S; i++) ret[i] = std::clamp(a[i], T(0), T(1));
        return ret;
    }
};

#define VEC_IMPL(ret, T) \
T dot(const ret<T>& b) const { return raw::dot(data, b.data); } \
T lengthSqr() const { return raw::lengthSqr(data); } \
T length() const { return raw::length(data); } \
ret<T> operator +(const ret<T>& b) const { return ret<T>(data + b.data); } \
ret<T> operator -(const ret<T>& b) const { return ret<T>(data - b.data); } \
ret<T> operator *(const ret<T>& b) const { return ret<T>(raw::operator*(data, b.data)); } \
ret<T> operator *(T scalar) const { return ret<T>(raw::operator*(data, scalar)); } \
ret<T> operator /(T scalar) const { return ret<T>(raw::operator/(data, scalar)); } \
ret<T> normalized() const { return ret<T>(raw::normalized(data)); } \
ret<T> lerp(const ret<T>& b, float t) const { return ret<T>(raw::lerp(data, b.data, t)); } \
ret<T> saturate() const { return ret<T>(raw::saturate(data)); } \
friend ret<T> operator *(T scalar, const ret<T>& vec) { return ret<T>(raw::operator*(vec.data, scalar)); } \
friend ret<T> operator /(T scalar, const ret<T>& vec) { return ret<T>(raw::operator/(scalar, vec.data)); } \
const T& operator [](size_t i) const { return data[i]; } \
T& operator [](size_t i) { return data[i]; }

class mat4;

namespace utils {
    template <typename T, size_t S>
    static void arrayAssign(std::array<T, S>& left, const std::array<T, S>& right) {
        for (size_t i = 0; i < S; i++) left[i] = right[i];
    }
}

template <typename T>
struct vec2 {
    vec2() = default;
    vec2(T x, T y) { utils::arrayAssign(data, { x, y }); }
    vec2(const std::array<T, 2>& val) { utils::arrayAssign(data, val); }

    union {
        struct { T x, y; };
        struct { T s, t; };
        std::array<T, 2> data{ T(0) };
    };

    VEC_IMPL(vec2, T)
};

template <typename T>
struct vec3 {
    vec3() = default;
    vec3(T x, T y, T z) { utils::arrayAssign(data, { x, y, z }); }
    vec3(const std::array<T, 3>& val) { utils::arrayAssign(data, val); }

    vec3<T> cross(const vec3<T>& b) const {
        vec3<T> ret{};
        ret[0] = this->data[1] * b[2] - this->data[2] * b[1];
        ret[1] = this->data[2] * b[0] - this->data[0] * b[2];
        ret[2] = this->data[0] * b[1] - this->data[1] * b[0];
        return ret;
    }

    union {
        struct { T x, y, z; };
        struct { T s, t, r; };
        std::array<T, 3> data;
    };

    VEC_IMPL(vec3, T)
};

template <typename T>
struct vec4 {
    vec4() = default;
    vec4(T x, T y, T z, T w = 1.0f) { utils::arrayAssign(data, { x, y, z, w }); }
    vec4(const std::array<T, 4>& val) { utils::arrayAssign(data, val); }
    vec4(const vec3<T>& v, T w = T(1)) { utils::arrayAssign(data, { v.x, v.y, v.z, w }); }

    vec3<T> xyz() const {
        return vec3<T>(x, y, z);
    }

    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
        std::array<T, 4> data;
    };

    VEC_IMPL(vec4, T)
};

typedef vec2<float> vec2f;
typedef vec3<float> vec3f;
typedef vec4<float> vec4f;

typedef vec2<int32_t> vec2i;
typedef vec3<int32_t> vec3i;
typedef vec4<int32_t> vec4i;

class mat4 {
public:
    mat4() {
        ::memset(m_rows, 0, sizeof(float) * 16);
        m_rows[0][0] = m_rows[1][1] = m_rows[2][2] = m_rows[3][3] = 1.0f;
    }

    mat4(const std::array<float, 16>& m) {
        ::memcpy(m_rows, m.data(), sizeof(float) * 16);
    }

    mat4 operator *(const mat4& mat) {
        mat4 ret{};
        __m128 out0x = lincomb_SSE(s_rows[0], mat);
        __m128 out1x = lincomb_SSE(s_rows[1], mat);
        __m128 out2x = lincomb_SSE(s_rows[2], mat);
        __m128 out3x = lincomb_SSE(s_rows[3], mat);
        ret.s_rows[0] = out0x;
        ret.s_rows[1] = out1x;
        ret.s_rows[2] = out2x;
        ret.s_rows[3] = out3x;
        return ret;
    }

    vec4f operator *(const vec4f& v) {
        const mat4& m = (*this);
        return vec4(
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
            m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3]
        );
    }

    vec4f operator *(const vec3f& b) {
        return ((*this) * vec4f(b[0], b[1], b[2], 1.0f));
    }

    static mat4 translation(const vec3f& t) {
        return mat4({
            1.0f, 0.0f, 0.0f, t.x,
            0.0f, 1.0f, 0.0f, t.y,
            0.0f, 0.0f, 1.0f, t.z,
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    static mat4 scaling(const vec3f& s) {
        return mat4({
            s.x, 0.0f, 0.0f, 0.0f,
            0.0f, s.y, 0.0f, 0.0f,
            0.0f, 0.0f, s.z, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    static mat4 rotationX(float angle) {
        float s = ::sinf(angle), c = ::cosf(angle);
        return mat4({
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, c, -s, 0.0f,
            0.0f, s, c, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    static mat4 rotationY(float angle) {
        float s = ::sinf(angle), c = ::cosf(angle);
        return mat4({
            c, 0.0f, s, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            -s, 0.0f, c, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    static mat4 rotationZ(float angle) {
        float s = ::sinf(angle), c = ::cosf(angle);
        return mat4({
            c, -s, 0.0f, 0.0f,
            s, c, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    static mat4 axisAngle(vec3f axis, float angle) {
        const float s = ::sinf(angle),
            c = ::cosf(angle),
            t = 1.0f - c;
        vec3f ax = axis.normalized();
        float x = ax.x, y = ax.y, z = ax.z;
        return mat4({
            t * x * x + c, t * x * y - z * s, t * x * z + y * s, 0,
            t * x * y + z * s, t * y * y + c, t * y * z + x * s, 0,
            t * x * z - y * s, t * y * z + x * s, t * z * z + c, 0,
            0, 0, 0, 1
        });
    }

    static mat4 orthographic(float left, float right, float bottom, float top, float near, float far) {
        const float w = right - left;
        const float h = top - bottom;
        const float d = far - near;
        return mat4({
            2.0f / w, 0.0f, 0.0f, -((right + left) / w),
            0.0f, 2.0f / h, 0.0f, -((top + bottom) / h),
            0.0f, 0.0f, -2.0f / d, -((far + near) / d),
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    static mat4 perspective(float fov, float aspect, float near, float far) {
        const float cotHalfFov = 1.0f / std::tanf(fov / 2.0f);
        return mat4({
            cotHalfFov / aspect, 0.0f, 0.0f, 0.0f,
            0.0f, cotHalfFov, 0.0f, 0.0f,
            0.0f, 0.0f, (near + far) / (near - far), (2.0f * near * far) / (near - far),
            0.0f, 0.0f, -1.0f, 0.0f
        });
    }

    static mat4 viewport(int x, int y, int w, int h) {
        float hw = w / 2.0f;
        float hh = h / 2.0f;
        return mat4({
            hw, 0.0f, 0.0f, x + hw,
            0.0f, -hh, 0.0f, y + hh,
            0.0f, 0.0f, 0.5f, 0.5f,
            0.0f, 0.0f, 0.0f, 1.0f
        });
    }

    mat4 transposed() const {
        mat4 res = (*this);
        _MM_TRANSPOSE4_PS(res.s_rows[0], res.s_rows[1], res.s_rows[2], res.s_rows[3]);
        return res;
    }

    mat4 inverted() {
        mat4 ret{};

        float* m = data();
        float inv[16], det;

        inv[0] = m[5] * m[10] * m[15] -
            m[5] * m[11] * m[14] -
            m[9] * m[6] * m[15] +
            m[9] * m[7] * m[14] +
            m[13] * m[6] * m[11] -
            m[13] * m[7] * m[10];

        inv[4] = -m[4] * m[10] * m[15] +
            m[4] * m[11] * m[14] +
            m[8] * m[6] * m[15] -
            m[8] * m[7] * m[14] -
            m[12] * m[6] * m[11] +
            m[12] * m[7] * m[10];

        inv[8] = m[4] * m[9] * m[15] -
            m[4] * m[11] * m[13] -
            m[8] * m[5] * m[15] +
            m[8] * m[7] * m[13] +
            m[12] * m[5] * m[11] -
            m[12] * m[7] * m[9];

        inv[12] = -m[4] * m[9] * m[14] +
            m[4] * m[10] * m[13] +
            m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

        inv[1] = -m[1] * m[10] * m[15] +
            m[1] * m[11] * m[14] +
            m[9] * m[2] * m[15] -
            m[9] * m[3] * m[14] -
            m[13] * m[2] * m[11] +
            m[13] * m[3] * m[10];

        inv[5] = m[0] * m[10] * m[15] -
            m[0] * m[11] * m[14] -
            m[8] * m[2] * m[15] +
            m[8] * m[3] * m[14] +
            m[12] * m[2] * m[11] -
            m[12] * m[3] * m[10];

        inv[9] = -m[0] * m[9] * m[15] +
            m[0] * m[11] * m[13] +
            m[8] * m[1] * m[15] -
            m[8] * m[3] * m[13] -
            m[12] * m[1] * m[11] +
            m[12] * m[3] * m[9];

        inv[13] = m[0] * m[9] * m[14] -
            m[0] * m[10] * m[13] -
            m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] -
            m[1] * m[7] * m[14] -
            m[5] * m[2] * m[15] +
            m[5] * m[3] * m[14] +
            m[13] * m[2] * m[7] -
            m[13] * m[3] * m[6];

        inv[6] = -m[0] * m[6] * m[15] +
            m[0] * m[7] * m[14] +
            m[4] * m[2] * m[15] -
            m[4] * m[3] * m[14] -
            m[12] * m[2] * m[7] +
            m[12] * m[3] * m[6];

        inv[10] = m[0] * m[5] * m[15] -
            m[0] * m[7] * m[13] -
            m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

        inv[14] = -m[0] * m[5] * m[14] +
            m[0] * m[6] * m[13] +
            m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] +
            m[1] * m[7] * m[10] +
            m[5] * m[2] * m[11] -
            m[5] * m[3] * m[10] -
            m[9] * m[2] * m[7] +
            m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] -
            m[0] * m[7] * m[10] -
            m[4] * m[2] * m[11] +
            m[4] * m[3] * m[10] +
            m[8] * m[2] * m[7] -
            m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if (det == 0)
            return (*this);

        det = 1.0f / det;

        for (int i = 0; i < 16; i++) ret.data()[i] = inv[i] * det;

        return ret;
    }

    float* operator [](size_t i) { return m_rows[i]; }
    const float* operator [](size_t i) const { return m_rows[i]; }

    inline float* data() { return &m_rows[0][0]; }

private:
    union {
        float m_rows[4][4];
        __m128 s_rows[4];
    };

    inline __m128 lincomb_SSE(const __m128& a, const mat4& b) {
        __m128 result;
        result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), b.s_rows[0]);
        result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x55), b.s_rows[1]));
        result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xaa), b.s_rows[2]));
        result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xff), b.s_rows[3]));
        return result;
    }
};

#endif // VECMATH_HPP
