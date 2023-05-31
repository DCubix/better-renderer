#include <iostream>

#define SDL_MAIN_HANDLED
#define RENDERER_BACKEND_SDL2
#include "renderer.hpp"

#define CUTE_PNG_IMPLEMENTATION
#include "cute_png.h"

#define TIME_STEP (1000 / 120)

#pragma region OBJParser
#include <fstream>
#include <sstream>
#include <string>

using Mesh = std::tuple<std::vector<VertexF>, std::vector<uint32_t>>;

static Mesh loadOBJ(const std::string& fileName) {
	using Face = std::array<int32_t, 3>;

	std::ifstream fp(fileName);
	if (fp.bad()) return { std::vector<VertexF>(), std::vector<uint32_t>() };

	std::vector<vec3f> vPositions;
	std::vector<vec3f> vNormals;
	std::vector<vec2f> vTexCoords;
	std::vector<Face> vRawIndices;

	std::string line;
	while (std::getline(fp, line)) {
		// skip comments
		if (line[0] == '#') continue;

		// skip empty or all-spaces string
		if (std::all_of(line.begin(), line.end(), ::isspace)) continue;

		std::istringstream iss(line);

		std::string type; iss >> type;

		auto fnParseFaceIndices = [](const std::string& src) {
			std::string token;
			std::istringstream ss(src);
			Face posTexNorm{ -1, -1, -1 };

			size_t index = 0;
			while (std::getline(ss, token, '/')) {
				posTexNorm[index++] = token.empty() ? -1 : std::stoi(token)-1;
			}

			return posTexNorm;
		};

		if (type == "v") { // positions
			vec3f pos;
			iss >> pos.x >> pos.y >> pos.z;
			vPositions.push_back(pos);
		}
		else if (type == "vn") { // normals
			vec3f norm;
			iss >> norm.x >> norm.y >> norm.z;
			vNormals.push_back(norm);
		}
		else if (type == "vt") { // texCoords
			vec2f uv;
			iss >> uv.x >> uv.y;
			vTexCoords.push_back(uv);
		}
		else if (type == "f") {
			// v, v/vt, v//vn, v/vt/vn
			for (size_t i = 0; i < 3; i++) {
				std::string src; iss >> src;
				vRawIndices.push_back(fnParseFaceIndices(src));
			}
		}
		else {
			continue;
		}
	}

	fp.close();

	std::vector<VertexF> retVertices;
	std::vector<uint32_t> retIndices;

	uint32_t index = 0;
	for (auto [pos, tex, norm] : vRawIndices) {
		VertexF vertex;
		vertex.position = vec4(vPositions[pos], 1.0f);
		vertex.normal = norm >= 0 ? vNormals[norm] : vec3f(0.0f, 0.0f, 0.0f);
		vertex.uv = tex >= 0 ? vTexCoords[tex] : vec2f(0.0f, 0.0f);
		vertex.color = vec4(1.0f, 1.0f, 1.0f, 1.0f);

		retVertices.push_back(vertex);
		retIndices.push_back(index++);
	}

	return { retVertices, retIndices };
}
#pragma endregion

class MonkeyPixelShader : public IPixelShader {
public:
	vec4f process(const TextureSlots& slots) {
		vec4f uv = inputs[Field::UV] * 4.0f;
		vec4f color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
		if (slots[0]) {
			vec4f texel = slots[0]->sampleFloat(uv.x, uv.y);
			if (texel.w <= 0.5f) {
				discard();
			}
			color = color * texel;
		}
		return color;
	}

	vec3f eye;
};

class CubePixelShader : public IPixelShader {
public:
	vec4f process(const TextureSlots& slots) {
		vec4f uv = inputs[Field::UV];
		vec4f color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
		if (slots[0]) {
			vec4f texel = slots[0]->sampleFloat(uv.x, uv.y);
			color = color * texel;
		}
		return color;
	}

};

int main() {


	SDL_Init(SDL_INIT_EVERYTHING);

	SDL_Window* window = SDL_CreateWindow(
		"Rasterizer",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		800, 600,
		SDL_WINDOW_SHOWN
	);
	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	int startTime = SDL_GetTicks();
	int accumulated = 0;

	std::unique_ptr<Renderer> ren = std::make_unique<Renderer>(new SDLRendererAdapter(renderer, 800, 600));
	SDLRendererAdapter* sdlRen = static_cast<SDLRendererAdapter*>(ren->adapter());

	// Demo
	auto [ monkeVertices, monkeIndices ] = loadOBJ("blob.obj");
	auto [ cubeVertices, cubeIndices ] = loadOBJ("cube_s.obj");

	cp_image_t img = cp_load_png("bricks.png");
	std::unique_ptr<Texture> texture = std::make_unique<Texture>(img.w, img.h, PixelFormat::RGBA);
	texture->load(reinterpret_cast<uint8_t*>(img.pix));
	free(img.pix);
	CUTE_PNG_MEMSET(&img, 0, sizeof(img));

	cp_image_t img2 = cp_load_png("metal.png");
	std::unique_ptr<Texture> metalTexture = std::make_unique<Texture>(img2.w, img2.h, PixelFormat::RGBA);
	metalTexture->load(reinterpret_cast<uint8_t*>(img2.pix));
	free(img2.pix);
	CUTE_PNG_MEMSET(&img2, 0, sizeof(img2));

	std::unique_ptr<DefaultVertexShader> vs = std::make_unique<DefaultVertexShader>(float(sdlRen->bufferWidth()) / sdlRen->bufferHeight());
	std::unique_ptr<MonkeyPixelShader> monkeyFs = std::make_unique<MonkeyPixelShader>();
	std::unique_ptr<CubePixelShader> cubeFs = std::make_unique<CubePixelShader>();
	std::unique_ptr<RenderTarget> renderTarget = std::make_unique<RenderTarget>(512, 512, PixelFormat::RGBA);

	float rot = 0.0f;
	mat4 cube_model = mat4();
	//

	int32_t frame = 0;
	int32_t frameTime = 0;

	bool running = true;
	SDL_Event event;
	while (running) {
		bool canDraw = false;

		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) running = false;
		}

		int currentTime = SDL_GetTicks();
		int delta = currentTime - startTime;
		startTime = currentTime;
		accumulated += delta;

		while (accumulated >= TIME_STEP) {
			accumulated -= TIME_STEP;
			// TODO: perform updates here
			rot += (1.0f / 60.0f);
			cube_model = mat4::rotationX(rot) * mat4::rotationY(rot);
			
			monkeyFs->eye = vec3f(
				vs->view[0][3],
				vs->view[1][3],
				vs->view[2][3]
			) * -1.0f;

			//fs->scale += (1.0f / 60.0f) * 0.5f;

			frameTime += TIME_STEP;
			if (frameTime >= 1000) {
				/*cp_image_t img;
				img.w = 512;
				img.h = 512;
				img.pix = reinterpret_cast<cp_pixel_t*>(renderTarget->colorTexture()->pixels());
				cp_save_png("out.png", &img);*/

				std::cout << frame << " fps\n";
				frameTime = 0;
				frame = 0;
			}

			canDraw = true;
		}

		if (canDraw) {
			
			vs->model = cube_model;
			vs->view = mat4::translation(vec3f(0.0f, 0.0f, -3.5f));
			//vs->projection = mat4::perspective(VM_DEG_TO_RAD(60.0f), 1.0f, 0.1f, 500.0f);
			vs->projection = mat4::perspective(VM_DEG_TO_RAD(60.0f), 800.0f / 600.0f, 0.1f, 500.0f);

			// To render target
			//ren->renderTarget(renderTarget.get());
			//ren->viewport(0, 0, 512, 512);
			ren->clear(vec4(0.3f, 0.1f, 0.0f, 1.0f));

			ren->bindTexture(texture.get(), 0);
			ren->bindTexture(metalTexture.get(), 1);

			ren->beginMesh(vs.get(), monkeyFs.get());
			ren->pushVertices(monkeVertices);
			ren->pushIndices(monkeIndices);
			ren->endMesh();

			//vs->view = mat4::translation(vec3f(0.0f, 0.0f, -4.5f));
			//vs->projection = mat4::perspective(VM_DEG_TO_RAD(60.0f), 800.0f/600.0f, 0.1f, 500.0f);

			// To screen
			//ren->renderTarget(nullptr);
			//ren->viewport(0, 0, 800, 600);
			//ren->clear(vec4(0.0f, 0.1f, 0.2f, 1.0f));

			//ren->bindTexture(renderTarget->colorTexture(), 0);

			//ren->beginMesh(vs.get(), cubeFs.get());
			//ren->pushVertices(cubeVertices);
			//ren->pushIndices(cubeIndices);
			//ren->endMesh();

			ren->swapBuffers();
			frame++;
		}
	}

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	return 0;
}
