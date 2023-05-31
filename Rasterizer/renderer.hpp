#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <map>
#include <memory>
#include <tuple>
#include <algorithm>
#include <functional>
#include <iostream>

#include "vecmath.hpp"

constexpr size_t RenderTileSize = 32;
const uint8_t DITHER_4x4[4][4] = {
	{ 0, 8, 2, 10},
	{12, 4, 14, 6},
	{ 3, 11, 1, 9},
	{15, 7, 13, 5}
};

namespace utils {
	static float findClosest(int32_t x, int32_t y, float value) {
		float limit = float(DITHER_4x4[y & 3][x & 3]) / 64.0f;
		return std::min(value + limit, 1.0f);
	}

	static vec3f findClosest(int32_t x, int32_t y, vec3f color) {
		return vec3f(
			findClosest(x, y, color.x),
			findClosest(x, y, color.y),
			findClosest(x, y, color.z)
		);
	}

	static uint32_t colorConvert(vec3f color) {
		uint8_t r = uint8_t(color.x * 255.0f);
		uint8_t g = uint8_t(color.y * 255.0f);
		uint8_t b = uint8_t(color.z * 255.0f);
		return ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
	}

	static float fastFloor(float x) {
		int xi = int(x);
		if (x < xi) return xi - 1; //Negative side rounds up
		return xi;
	}

	static float fract(float x) {
		return x - fastFloor(x);
	}

	static float modulo(float x, float n) {
		return fract(x / n) * n;
	}
}

struct VertexF {
	vec4f position;
	vec3f normal;
	vec4f color;
	vec2f uv;

	bool insideFustrum() {
		return ::abs(position.x) <= ::abs(position.w) &&
			::abs(position.y) <= ::abs(position.w) &&
			::abs(position.z) <= ::abs(position.w);
	}

	VertexF lerp(const VertexF& b, float factor) const {
		return {
			.position = position.lerp(b.position, factor),
			.normal = normal.lerp(b.normal, factor),
			.color = color.lerp(b.color, factor),
			.uv = uv.lerp(b.uv, factor)
		};
	}
};

struct VertexI {
	vec2i position;
	vec4f worldPosition;
	vec3f normal;
	vec4f color;
	vec2f uv;
};

enum class PixelFormat : uint8_t {
	SINGLE = 1,
	RG,
	RGB,
	RGBA,
	DEPTH
};

class Texture {
	using Pixel = std::array<uint8_t, 4>;
public:
	Texture() = default;
	Texture(uint32_t width, uint32_t height, PixelFormat format)
		: m_width(width), m_height(height), m_format(format)
	{
		createPixels();
	}

	~Texture() {
		delete[] m_pixels;
		m_pixels = nullptr;
	}

	uint32_t width() const { return m_width; }
	uint32_t height() const { return m_height; }
	PixelFormat format() const { return m_format; }

	void load(const uint8_t* data) {
		const size_t pixelCount = uint8_t(m_format) * m_width * m_height;
		for (size_t i = 0; i < pixelCount; i++) {
			m_pixels[i] = data[i];
		}
	}

	void set(int32_t x, int32_t y, const Pixel& color) {
		if (!m_pixels) return;
		if (x < 0 || y < 0 || x >= m_width || y >= m_height) return;
		setPixel(x, y, color);
	}

	void setf(int32_t x, int32_t y, const vec4f& color) {
		vec4f col = color.saturate();
		set(x, y, {
			uint8_t(col[0] * 255.0f),
			uint8_t(col[1] * 255.0f),
			uint8_t(col[2] * 255.0f),
			uint8_t(col[3] * 255.0f)
		});
	}

	Pixel sample(int32_t x, int32_t y) {
		if (!m_pixels) return { 0, 0, 0, 0 };
		if (x < 0 || y < 0 || x >= m_width || y >= m_height) return { 0, 0, 0, 0 };
		return getPixel(x, y);
	}

	Pixel sample(float x, float y) {
		x = utils::fract(x);
		y = utils::fract(y);

		return sample(
			int32_t(x * float(m_width - 1) + 0.5f),
			int32_t(y * float(m_height - 1) + 0.5f)
		);
	}

	float sampleDepth(int32_t x, int32_t y) {
		if (!m_pixels) return 0.0f;
		if (x < 0 || y < 0 || x >= m_width || y >= m_height) return 0.0f;
		return getDepth(x, y);
	}

	float sampleDepth(float x, float y) {
		x = utils::fract(x);
		y = utils::fract(y);

		return sampleDepth(
			int32_t(x * float(m_width - 1) + 0.5f),
			int32_t(y * float(m_height - 1) + 0.5f)
		);
	}

	vec4f sampleFloat(float x, float y) {
		auto color = sample(x, y);
		return vec4f(
			float(color[0]) / 255.0f,
			float(color[1]) / 255.0f,
			float(color[2]) / 255.0f,
			float(color[3]) / 255.0f
		);
	}

	vec4f sampleFloatBilinear(float x, float y) {
		const float resolutionX = m_width;
		const float resolutionY = m_height;

		x = x * resolutionX - 0.5f;
		y = y * resolutionY - 0.5f;

		float fu = utils::fract(x), fv = utils::fract(y);
		float iu = floorf(x), iv = floorf(y);

		vec4f c0 = sampleFloat((iu + 0.5f) / resolutionX, (iv + 0.5f) / resolutionY);
		vec4f c1 = sampleFloat((iu + 1.5f) / resolutionX, (iv + 0.5f) / resolutionY);
		vec4f c2 = sampleFloat((iu + 0.5f) / resolutionX, (iv + 1.5f) / resolutionY);
		vec4f c3 = sampleFloat((iu + 1.5f) / resolutionX, (iv + 1.5f) / resolutionY);

		vec4f c0c1 = c0.lerp(c1, fu);
		vec4f c2c3 = c2.lerp(c3, fu);

		return c0c1.lerp(c2c3, fv);
	}

	void setDepth(int32_t x, int32_t y, float value) {
		const size_t multiplier = sizeof(float);
		uint32_t offset = (x + y * m_width) * multiplier;
		::memcpy(m_pixels + offset, &value, sizeof(float));
	}

	uint8_t* pixels() { return m_pixels; }

private:
	uint32_t m_width, m_height;
	PixelFormat m_format;
	uint8_t* m_pixels{ nullptr };

	void createPixels() {
		switch (m_format) {
			case PixelFormat::SINGLE: m_pixels = new uint8_t[m_width * m_height]; break;
			case PixelFormat::RG: m_pixels = new uint8_t[m_width * m_height * 2]; break;
			case PixelFormat::RGB: m_pixels = new uint8_t[m_width * m_height * 3]; break;
			case PixelFormat::RGBA: m_pixels = new uint8_t[m_width * m_height * 4]; break;
			case PixelFormat::DEPTH: m_pixels = new uint8_t[m_width * m_height * sizeof(float)]; break;
		}
	}

	Pixel getPixel(int32_t x, int32_t y) {
		switch (m_format) {
			case PixelFormat::SINGLE: return { getComponent(x, y, 0) };
			case PixelFormat::RG: return { getComponent(x, y, 0), getComponent(x, y, 1) };
			case PixelFormat::RGB: return { getComponent(x, y, 0), getComponent(x, y, 1), getComponent(x, y, 2) };
			case PixelFormat::RGBA: return { getComponent(x, y, 0), getComponent(x, y, 1), getComponent(x, y, 2), getComponent(x, y, 3) };
			case PixelFormat::DEPTH: {
				float depth = getDepth(x, y) * 0.5f + 0.5f;
				uint32_t depth32 = uint32_t(depth * float(UINT32_MAX-1));
				Pixel p{}; ::memcpy(p.data(), &depth32, sizeof(uint32_t));
				return p;
			}
		}
	}

	void setPixel(int32_t x, int32_t y, const Pixel& value) {
		switch (m_format) {
			case PixelFormat::SINGLE: setComponent(x, y, 0, value[0]); break;
			case PixelFormat::RG:
				setComponent(x, y, 0, value[0]);
				setComponent(x, y, 1, value[1]);
				break;
			case PixelFormat::RGB:
				setComponent(x, y, 0, value[0]);
				setComponent(x, y, 1, value[1]);
				setComponent(x, y, 2, value[2]);
				break;
			case PixelFormat::RGBA:
				setComponent(x, y, 0, value[0]);
				setComponent(x, y, 1, value[1]);
				setComponent(x, y, 2, value[2]);
				setComponent(x, y, 3, value[3]);
				break;
			default: return;
		}
	}

	uint8_t getComponent(int32_t x, int32_t y, uint8_t index) {
		size_t multiplier = 1;
		switch (m_format) {
			case PixelFormat::SINGLE: multiplier = 1; break;
			case PixelFormat::RG: multiplier = 2; break;
			case PixelFormat::RGB: multiplier = 3; break;
			case PixelFormat::RGBA: multiplier = 4; break;
			default: return 0;
		}
		uint32_t offset = (x + y * m_width) * multiplier;
		return m_pixels[offset + index];
	}

	void setComponent(int32_t x, int32_t y, uint8_t index, uint8_t value) {
		size_t multiplier = 1;
		switch (m_format) {
			case PixelFormat::SINGLE: multiplier = 1; break;
			case PixelFormat::RG: multiplier = 2; break;
			case PixelFormat::RGB: multiplier = 3; break;
			case PixelFormat::RGBA: multiplier = 4; break;
		}
		uint32_t offset = (x + y * m_width) * multiplier;
		m_pixels[offset + index] = value;
	}

	float getDepth(int32_t x, int32_t y) {
		uint32_t offset = (x + y * m_width) * sizeof(float);

		union {
			float value;
			uint8_t raw[4];
		};
		raw[0] = m_pixels[offset + 0];
		raw[1] = m_pixels[offset + 1];
		raw[2] = m_pixels[offset + 2];
		raw[3] = m_pixels[offset + 3];

		return value;
	}
};

using TextureSlots = std::array<Texture*, 8>;

class RenderTarget {
public:
	RenderTarget() = default;
	RenderTarget(uint32_t width, uint32_t height, PixelFormat format)
	{
		m_colorTexture = std::make_unique<Texture>(width, height, format);
		m_depthBuffer = std::make_unique<Texture>(width, height, PixelFormat::DEPTH);
	}

	Texture* colorTexture() { return m_colorTexture.get(); }
	Texture* depthTexture() { return m_depthBuffer.get(); }

	void clearColor(vec4f color) {
		for (uint32_t y = 0; y < m_colorTexture->height(); y++) {
			for (uint32_t x = 0; x < m_colorTexture->width(); x++) {
				m_colorTexture->setf(x, y, color);
			}
		}
	}

	void clearDepth(float value = 1.0f) {
		for (uint32_t y = 0; y < m_depthBuffer->height(); y++) {
			for (uint32_t x = 0; x < m_depthBuffer->width(); x++) {
				m_depthBuffer->setDepth(x, y, value);
			}
		}
	}

protected:
	std::unique_ptr<Texture> m_colorTexture;
	std::unique_ptr<Texture> m_depthBuffer;
};

class IVertexShader {
public:
	enum Field {
		POSITION = 0,
		NORMAL,
		UV,
		COLOR,
		COUNT
	};

	virtual void process() = 0;

	std::array<vec4f, Field::COUNT> inputs;
	std::array<vec4f, Field::COUNT> outputs;
};

class Renderer;
class IPixelShader {
	friend class Renderer;
public:
	enum Field {
		POSITION = 0,
		NORMAL,
		UV,
		COLOR,
		COUNT
	};

	virtual vec4f process(const TextureSlots& slots) = 0;

	void discard() { m_discard = true; }

	std::array<vec4f, Field::COUNT> inputs;

protected:
	bool m_discard{ false };
};

class IRendererAdapter {
public:
	virtual void blit(size_t pixelsSize, const uint8_t* pixels) = 0;

	uint32_t bufferWidth() const { return m_bufferWidth; }
	uint32_t bufferHeight() const { return m_bufferHeight; }

protected:
	uint32_t m_bufferWidth, m_bufferHeight;
};

class Renderer {
public:
	enum CullMode {
		CullBack = 0,
		CullFront
	};

	Renderer() = default;
	Renderer(IRendererAdapter* adapter) {
		m_adapter = std::unique_ptr<IRendererAdapter>(adapter);

		m_defaultTarget = std::make_unique<RenderTarget>(
			m_adapter->bufferWidth(),
			m_adapter->bufferHeight(),
			PixelFormat::RGB
		);

		m_currentTarget = m_defaultTarget.get();

		m_viewport[0] = 0;
		m_viewport[1] = 0;
		m_viewport[2] = m_adapter->bufferWidth();
		m_viewport[3] = m_adapter->bufferHeight();
	}

	IRendererAdapter* adapter() { return m_adapter.get(); }

	void cullMode(CullMode mode) { m_cullMode = mode; }
	CullMode cullMode(CullMode mode) const { return m_cullMode; }

	void bindTexture(Texture* texture, size_t slot = 0) { m_textureSlots[slot] = texture; }
	void renderTarget(RenderTarget* target) { m_currentTarget = target; }
	void viewport(int32_t x, int32_t y, int32_t width, int32_t height) {
		m_viewport[0] = x;
		m_viewport[1] = y;
		m_viewport[2] = width;
		m_viewport[3] = height;
	}

	void clear(vec4f color, float depth = 1.0f) {
		if (!m_currentTarget) m_currentTarget = m_defaultTarget.get();

		m_currentTarget->clearColor(color);
		m_currentTarget->clearDepth(depth);
	}

	void drawPixel(int32_t x, int32_t y, vec4f color) {
		if (!m_currentTarget) m_currentTarget = m_defaultTarget.get();
		m_currentTarget->colorTexture()->setf(x, y, color);
	}

	void swapBuffers() {
		if (!m_currentTarget) m_currentTarget = m_defaultTarget.get();

		m_adapter->blit(
			m_currentTarget->colorTexture()->width() *
			m_currentTarget->colorTexture()->height() *
			size_t(m_currentTarget->colorTexture()->format()),

			m_currentTarget->colorTexture()->pixels()
		);
	}

	void beginMesh(IVertexShader* vertexShader, IPixelShader* pixelShader) {
		if (m_drawing) return;
		m_drawing = true;

		m_vertexShader = vertexShader;
		m_pixelShader = pixelShader;
	}

	void pushVertex(VertexF vertex) {
		if (!m_drawing) return;
		m_currentVertices.push_back(vertex);
	}

	void pushVertices(const std::vector<VertexF>& vertices) {
		if (!m_drawing) return;
		for (auto& vertex : vertices) pushVertex(vertex);
	}

	void pushIndex(uint32_t index) {
		if (!m_drawing) return;
		m_currentIndices.push_back(index);
	}

	void pushIndices(const std::vector<uint32_t>& indices) {
		if (!m_drawing) return;
		for (auto& index : indices) pushIndex(index);
	}

	void endMesh() {
		if (!m_drawing) return;
		m_drawing = false;

		if (m_currentIndices.size() < 3) { // do not allow anything other than triangles ;)
			m_currentVertices.clear();
			m_currentIndices.clear();
			return;
		}

		for (auto&& vertex : m_currentVertices) {
			transformVertex(vertex);
		}

		// Draw the triangles
		for (int i = 0; i < m_currentIndices.size(); i += 3) {
			uint32_t i0 = m_currentIndices[i + 0];
			uint32_t i1 = m_currentIndices[i + 1];
			uint32_t i2 = m_currentIndices[i + 2];

			// TODO: Backface culling, how do I get the view direction here?

			drawClippedTriangle(
				m_currentVertices[i0],
				m_currentVertices[i1],
				m_currentVertices[i2]
			);
		}

		m_currentVertices.clear();
		m_currentIndices.clear();
	}
	
private:
	std::unique_ptr<IRendererAdapter> m_adapter;

	// Drawing/Rendering
	bool m_drawing{ false };
	std::vector<VertexF> m_currentVertices;
	std::vector<uint32_t> m_currentIndices;
	
	IVertexShader* m_vertexShader{ nullptr };
	IPixelShader* m_pixelShader{ nullptr };

	// Render states
	std::unique_ptr<RenderTarget> m_defaultTarget;
	RenderTarget* m_currentTarget{ nullptr };
	
	std::array<int32_t, 4> m_viewport;
	CullMode m_cullMode{ CullMode::CullBack };
	TextureSlots m_textureSlots{ nullptr };
	//

	class Slope {
	public:
		Slope() = default;
		Slope(float begin, float end, int32_t numSteps) {
			const float invStep = 1.0f / numSteps;
			m_current = begin;
			m_step = (end - begin) * invStep;
			m_numSteps = numSteps;
		}

		float get() { return m_current; }
		void advance() { m_current += m_step; }
		int32_t numSteps() { return m_numSteps; }

	private:
		float m_current{ 0.0f }, m_step{ 0.0f };
		int32_t m_numSteps;
	};

	enum SlopeIndex {
		SLOPE_X = 0,
		SLOPE_R,
		SLOPE_G,
		SLOPE_B,
		SLOPE_A,
		SLOPE_U,
		SLOPE_V,
		SLOPE_Z,
		SLOPE_W,

		SLOPE_WORLD_X,
		SLOPE_WORLD_Y,
		SLOPE_WORLD_Z,

		SLOPE_NORMAL_X,
		SLOPE_NORMAL_Y,
		SLOPE_NORMAL_Z,

		SLOPE_INDEX_COUNT
	};
	using SlopeData = std::array<Slope, SLOPE_INDEX_COUNT>;

	using FNMakeSlope = std::function<SlopeData(VertexI, VertexI, int32_t)>;
	using FNScanline = std::function<void(int32_t, SlopeData&, SlopeData&)>;

	void transformVertex(VertexF& vertex) {
		if (!m_vertexShader) {
			return;
		}

		m_vertexShader->inputs[IVertexShader::POSITION] = vertex.position;
		m_vertexShader->inputs[IVertexShader::NORMAL] = vec4f(vertex.normal, 0.0f);
		m_vertexShader->inputs[IVertexShader::UV] = vec4f(vertex.uv.x, vertex.uv.y, 0.0f, 0.0f);
		m_vertexShader->inputs[IVertexShader::COLOR] = vertex.color;

		m_vertexShader->process();

		vertex.position = m_vertexShader->outputs[IVertexShader::POSITION];
		vertex.normal = m_vertexShader->outputs[IVertexShader::NORMAL].xyz();
		vertex.uv.x = m_vertexShader->outputs[IVertexShader::UV][0];
		vertex.uv.y = m_vertexShader->outputs[IVertexShader::UV][1];
		vertex.color = m_vertexShader->outputs[IVertexShader::COLOR];
	}

	VertexI convertVertex(VertexF vertex) {
		vec4f position = vertex.position;
		float wComponent = position.w;

		// Perspective divide
		position = position / wComponent;

		// Viewport transform
		mat4 viewportMatrix = mat4::viewport(
			m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]
		);
		position = viewportMatrix * position;

		VertexI vi;
		vi.position.x = int32_t(position.x);
		vi.position.y = int32_t(position.y);
		vi.worldPosition = vec4f(position.xyz(), wComponent);
		vi.uv = vertex.uv;
		vi.color = vertex.color;
		vi.normal = vertex.normal;

		return vi;
	}

	void drawTriangle(
		VertexF v0, VertexF v1, VertexF v2,
		IPixelShader* pixelShader = nullptr
	) {
		if (!m_currentTarget) m_currentTarget = m_defaultTarget.get();

		FNMakeSlope makeSlope = [&](VertexI from, VertexI to, int32_t numSteps) {
			SlopeData result;

			float wBegin = 1.0f / from.worldPosition.w, wEnd = 1.0f / to.worldPosition.w;

			result[SLOPE_X] = Slope(float(from.position[0]), float(to.position[0]), numSteps);

			result[SLOPE_R] = Slope(from.color[0] * wBegin, to.color[0] * wEnd, numSteps);
			result[SLOPE_G] = Slope(from.color[1] * wBegin, to.color[1] * wEnd, numSteps);
			result[SLOPE_B] = Slope(from.color[2] * wBegin, to.color[2] * wEnd, numSteps);
			result[SLOPE_A] = Slope(from.color[3] * wBegin, to.color[3] * wEnd, numSteps);

			result[SLOPE_U] = Slope(from.uv[0] * wBegin, to.uv[0] * wEnd, numSteps);
			result[SLOPE_V] = Slope(from.uv[1] * wBegin, to.uv[1] * wEnd, numSteps);

			result[SLOPE_WORLD_X] = Slope(from.worldPosition[0] * wBegin, to.worldPosition[0] * wEnd, numSteps);
			result[SLOPE_WORLD_Y] = Slope(from.worldPosition[1] * wBegin, to.worldPosition[1] * wEnd, numSteps);
			result[SLOPE_WORLD_Z] = Slope(from.worldPosition[2] * wBegin, to.worldPosition[2] * wEnd, numSteps);

			result[SLOPE_NORMAL_X] = Slope(from.normal[0] * wBegin, to.normal[0] * wEnd, numSteps);
			result[SLOPE_NORMAL_Y] = Slope(from.normal[1] * wBegin, to.normal[1] * wEnd, numSteps);
			result[SLOPE_NORMAL_Z] = Slope(from.normal[2] * wBegin, to.normal[2] * wEnd, numSteps);

			result[SLOPE_Z] = Slope(from.worldPosition.z, to.worldPosition.z, numSteps);
			result[SLOPE_W] = Slope(wBegin, wEnd, numSteps);

			return result;
		};

		FNScanline scanline = [&](int32_t y, SlopeData& left, SlopeData& right) {
			int32_t x = left[SLOPE_X].get(), endX = right[SLOPE_X].get();
			int32_t numSteps = endX - x;

			Slope colorSlopes[4];
			for (size_t i = 0; i < 4; i++) {
				colorSlopes[i] = Slope(left[SLOPE_R + i].get(), right[SLOPE_R + i].get(), numSteps);
			}

			Slope uvSlopes[2];
			for (size_t i = 0; i < 2; i++) {
				uvSlopes[i] = Slope(left[SLOPE_U + i].get(), right[SLOPE_U + i].get(), numSteps);
			}

			Slope zSlope = Slope(left[SLOPE_Z].get(), right[SLOPE_Z].get(), numSteps);
			Slope wSlope = Slope(left[SLOPE_W].get(), right[SLOPE_W].get(), numSteps);

			Slope worldSlopes[3];
			for (size_t i = 0; i < 3; i++) {
				worldSlopes[i] = Slope(left[SLOPE_WORLD_X + i].get(), right[SLOPE_WORLD_X + i].get(), numSteps);
			}

			Slope normalSlopes[3];
			for (size_t i = 0; i < 3; i++) {
				normalSlopes[i] = Slope(left[SLOPE_NORMAL_X + i].get(), right[SLOPE_NORMAL_X + i].get(), numSteps);
			}

			for (; x < endX; x++) {
				const size_t zBufferIndex = x + y * m_adapter->bufferWidth();
				if (zBufferIndex >= m_adapter->bufferWidth() * m_adapter->bufferHeight()) continue;

				float z = zSlope.get();
				float oldZ = m_currentTarget->depthTexture()->sampleDepth(x, y);
				float w = 1.0f / wSlope.get();

				if (oldZ > z) {
					if (!pixelShader) { // fallback basic rendering
						bool discard = false;

						vec3f color(
							colorSlopes[0].get() * w,
							colorSlopes[1].get() * w,
							colorSlopes[2].get() * w
						);

						if (m_textureSlots[0]) {
							float u = uvSlopes[0].get() * w;
							float v = uvSlopes[1].get() * w;
							vec4f texel = m_textureSlots[0]->sampleFloat(u, v);
							if (texel.w < 0.5f) {
								discard = true;
								z = 1.0f;
							}
							else {
								color = color * texel.xyz();
							}
						}

						if (!discard) drawPixel(x, y, color);
					}
					else {
						pixelShader->inputs[IPixelShader::UV][0] = uvSlopes[0].get() * w;
						pixelShader->inputs[IPixelShader::UV][1] = uvSlopes[1].get() * w;
						pixelShader->inputs[IPixelShader::UV][2] = 0.0f;
						pixelShader->inputs[IPixelShader::UV][3] = 0.0f;
						pixelShader->inputs[IPixelShader::COLOR][0] = colorSlopes[0].get() * w;
						pixelShader->inputs[IPixelShader::COLOR][1] = colorSlopes[1].get() * w;
						pixelShader->inputs[IPixelShader::COLOR][2] = colorSlopes[2].get() * w;
						pixelShader->inputs[IPixelShader::COLOR][3] = colorSlopes[3].get() * w;
						pixelShader->inputs[IPixelShader::POSITION][0] = worldSlopes[0].get() * w;
						pixelShader->inputs[IPixelShader::POSITION][1] = worldSlopes[1].get() * w;
						pixelShader->inputs[IPixelShader::POSITION][2] = worldSlopes[2].get() * w;
						pixelShader->inputs[IPixelShader::POSITION][3] = w;
						pixelShader->inputs[IPixelShader::NORMAL][0] = normalSlopes[0].get() * w;
						pixelShader->inputs[IPixelShader::NORMAL][1] = normalSlopes[1].get() * w;
						pixelShader->inputs[IPixelShader::NORMAL][2] = normalSlopes[2].get() * w;
						pixelShader->inputs[IPixelShader::NORMAL][3] = 0.0f;

						vec4f color = pixelShader->process(m_textureSlots).saturate();
						if (!pixelShader->m_discard) drawPixel(x, y, color);
						else z = 1.0f;

						pixelShader->m_discard = false;
					}

					m_currentTarget->depthTexture()->setDepth(x, y, z);
				}

				for (auto& slope : colorSlopes) slope.advance();
				for (auto& slope : uvSlopes) slope.advance();
				for (auto& slope : worldSlopes) slope.advance();
				for (auto& slope : normalSlopes) slope.advance();
				zSlope.advance();
				wSlope.advance();
			}

			// update x coords.
			for (auto& slope : left) slope.advance();
			for (auto& slope : right) slope.advance();
		};

		vec3f p0 = v0.position.xyz();
		vec3f p1 = v1.position.xyz();
		vec3f p2 = v2.position.xyz();

		bool cullOut = m_cullMode == CullBack ?
			p1.cross(p2).dot(p0) <= 1e-4f :
			p1.cross(p2).dot(p0) > 1e-4f;

		if (cullOut) {
			return;
		}

		rasterizeTriangle(
			convertVertex(v0),
			convertVertex(v1),
			convertVertex(v2),
			makeSlope, scanline
		);
	}

	void drawClippedTriangle(VertexF v0, VertexF v1, VertexF v2) {
		if (v0.insideFustrum() && v1.insideFustrum() && v2.insideFustrum()) {
			drawTriangle(v0, v1, v2, m_pixelShader);
			return;
		}

		std::vector<VertexF> vertices{ v0, v1, v2 }, aux;
		std::vector<VertexF> result;

		if (clipPolyAxis(vertices, aux, 0) &&
			clipPolyAxis(vertices, aux, 1) &&
			clipPolyAxis(vertices, aux, 2))
		{
			VertexF initial = vertices[0];
			for (size_t i = 1; i < vertices.size() - 1; i++) {
				VertexF v1 = vertices[i];
				VertexF v2 = vertices[i + 1];
				drawTriangle(initial, v1, v2, m_pixelShader);
			}
		}
	}

	bool clipPolyAxis(std::vector<VertexF>& vertices, std::vector<VertexF>& aux, size_t comp) {
		clipPolyComponent(vertices, comp, 1.0f, aux);
		vertices.clear();

		if (aux.empty()) {
			return false;
		}

		clipPolyComponent(aux, comp, -1.0f, vertices);
		aux.clear();

		return !vertices.empty();
	}

	void clipPolyComponent(const std::vector<VertexF>& vertices, size_t comp, float factor, std::vector<VertexF>& result) {
		VertexF prevVertex = vertices[vertices.size()-1];
		float prevComp = prevVertex.position[comp] * factor;
		bool prevInside = prevComp <= prevVertex.position.w;

		for (const VertexF& currVertex : vertices) {
			float currComp = currVertex.position[comp] * factor;
			bool currInside = currComp <= currVertex.position.w;

			if (currInside ^ prevInside) {
				float wMB = prevVertex.position.w - prevComp;
				float lerpFac = wMB / (wMB - (currVertex.position.w - currComp));

				VertexF b = prevVertex.lerp(currVertex, lerpFac);
				result.push_back(b);
			}

			if (currInside) result.push_back(currVertex);

			prevComp = currComp;
			prevInside = currInside;
			prevVertex = currVertex;
		}
	}

	void rasterizeTriangle(
		VertexI v0, VertexI v1, VertexI v2,
		FNMakeSlope fnMakeSlope,
		FNScanline fnDrawScanline
	) {
		int32_t x0 = v0.position[0], y0 = v0.position[1],
			x1 = v1.position[0], y1 = v1.position[1],
			x2 = v2.position[0], y2 = v2.position[1];

		if (std::tie(y1, x1) < std::tie(y0, x0)) { std::swap(x0, x1); std::swap(y0, y1); std::swap(v0, v1); }
		if (std::tie(y2, x2) < std::tie(y0, x0)) { std::swap(x0, x2); std::swap(y0, y2); std::swap(v0, v2); }
		if (std::tie(y2, x2) < std::tie(y1, x1)) { std::swap(x1, x2); std::swap(y1, y2); std::swap(v1, v2); }

		if (y0 == y2) return;

		bool shortSide = (y1 - y0) * (x2 - x0) < (x1 - x0) * (y2 - y0); // false=left, true=right

		SlopeData sides[2];
		sides[!shortSide] = fnMakeSlope(v0, v2, y2 - y0);

		for (int32_t y = y0, endY = y0; ; y++) {
			if (y >= endY) {
				if (y >= y2) break;

				if (y < y1) {
					sides[shortSide] = fnMakeSlope(v0, v1, y1 - y0);
					endY = y1;
				}
				else {
					sides[shortSide] = fnMakeSlope(v1, v2, y2 - y1);
					endY = y2;
				}
			}
			fnDrawScanline(y, sides[0], sides[1]);
		}
	}

};

class DefaultVertexShader : public IVertexShader {
public:
	DefaultVertexShader() = default;
	DefaultVertexShader(float aspect) {
		projection = mat4::perspective(VM_DEG_TO_RAD(60.0f), aspect, 0.1f, 500.0f);
		view = mat4::translation(vec3f({ 0.0f, 0.0f, -6.0f }));
	}

	void process() {
		outputs[Field::POSITION] = projection * view * model * inputs[Field::POSITION];
		outputs[Field::COLOR] = inputs[Field::COLOR];
		outputs[Field::UV] = inputs[Field::UV];
		outputs[Field::NORMAL] = view * model * inputs[Field::NORMAL];
	}

	mat4 projection{};
	mat4 model{};
	mat4 view{};
};

#if defined(RENDERER_BACKEND_SDL2)
#include <SDL2/SDL.h>

class SDLRendererAdapter : public IRendererAdapter {
public:
	SDLRendererAdapter(
		SDL_Renderer* renderer,
		uint32_t bufferWidth,
		uint32_t bufferHeight
	) : m_renderer(renderer)
	{
		m_buffer = SDL_CreateTexture(
			renderer,
			SDL_PIXELFORMAT_RGB24,
			SDL_TEXTUREACCESS_STREAMING,
			bufferWidth, bufferHeight
		);
		m_bufferWidth = bufferWidth;
		m_bufferHeight = bufferHeight;
	}

	~SDLRendererAdapter() {
		SDL_DestroyTexture(m_buffer);
	}

	void blit(size_t pixelsSize, const uint8_t* pixels) {
		SDL_RenderClear(m_renderer);
		
		void* writableBuffer;
		int pitch;
		SDL_LockTexture(m_buffer, NULL, &writableBuffer, &pitch);
		::memcpy(writableBuffer, pixels, pitch * m_bufferHeight);
		SDL_UnlockTexture(m_buffer);

		SDL_RenderCopy(m_renderer, m_buffer, NULL, NULL);
		SDL_RenderPresent(m_renderer);
	}

private:
	SDL_Texture* m_buffer;
	SDL_Renderer* m_renderer;
};
#endif

#endif // RENDERER_HPP
