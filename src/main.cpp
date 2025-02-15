// For assert macro
#include <cassert>

// SDL 3 header
#include <SDL3/SDL.h>

// GLM configuration and headers
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // GLM clip space should be in Z-axis to 0 to 1
#define GLM_FORCE_LEFT_HANDED       // GLM should use left-handed coordinates, +z goes into screen
#define GLM_FORCE_RADIANS           // GLM should always use radians not degrees.
#include <glm/glm.hpp>              // Required for glm::vec3/4/mat4/etc
#include <glm/ext.hpp>              // Required for glm::perspective function

import std;
import colors;
import io;

using namespace std::literals;

constexpr auto IS_DEBUG       = bool{ _DEBUG };
constexpr auto DEPTH_FORMAT   = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;
constexpr auto MAX_ANISOTROPY = float{ 16 };

namespace sdl3
{
	// Deleters for SDL objects
	template <auto fn>
	struct sdl_deleter
	{
		constexpr void operator()(auto *arg)
		{
			fn(arg);
		}
	};
	// Define SDL types with std::unique_ptr and custom deleter;
	using gpu_ptr    = std::unique_ptr<SDL_GPUDevice, sdl_deleter<SDL_DestroyGPUDevice>>;
	using window_ptr = std::unique_ptr<SDL_Window, sdl_deleter<SDL_DestroyWindow>>;

	template <auto fn>
	struct gpu_deleter
	{
		SDL_GPUDevice *gpu = nullptr;
		constexpr void operator()(auto *arg)
		{
			fn(gpu, arg);
		}
	};
	// Typedefs for SDL objects that need GPU Device to properly destruct
	using free_gfx_pipeline = gpu_deleter<SDL_ReleaseGPUGraphicsPipeline>;
	using gfx_pipeline_ptr  = std::unique_ptr<SDL_GPUGraphicsPipeline, free_gfx_pipeline>;
	using free_gfx_shader   = gpu_deleter<SDL_ReleaseGPUShader>;
	using gpu_shader_ptr    = std::unique_ptr<SDL_GPUShader, free_gfx_shader>;
	using free_buffer       = gpu_deleter<SDL_ReleaseGPUBuffer>;
	using gpu_buffer_ptr    = std::unique_ptr<SDL_GPUBuffer, free_buffer>;
	using free_texture      = gpu_deleter<SDL_ReleaseGPUTexture>;
	using gpu_texture_ptr   = std::unique_ptr<SDL_GPUTexture, free_texture>;
	using free_sampler      = gpu_deleter<SDL_ReleaseGPUSampler>;
	using gpu_sampler_ptr   = std::unique_ptr<SDL_GPUSampler, free_sampler>;

	struct context
	{
		window_ptr window;
		gpu_ptr gpu;
	};

	auto init_context(uint32_t width, uint32_t height, std::string_view title) -> context
	{
		std::println("{}Initialize SDL, GPU and create window.{}", colors::BLU, colors::RESET);

		auto result = SDL_Init(SDL_INIT_VIDEO);
		assert(result == true && "SDL could not be initialize.");

		auto w      = static_cast<int>(width);
		auto h      = static_cast<int>(height);
		auto window = SDL_CreateWindow(title.data(), w, h, NULL);
		assert(window != nullptr && "Failed to create window.");

		auto gpu = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_DXIL, IS_DEBUG, NULL);
		assert(gpu != nullptr && "Could not get a GPU device.");

		result = SDL_ClaimWindowForGPUDevice(gpu, window);
		assert(result == true && "Could not claim window for gpu.");

		return {
			.window = window_ptr(window),
			.gpu    = gpu_ptr(gpu),
		};
	}

	auto destroy_context(context &ctx)
	{
		std::println("{}Destroy SDL context.{}", colors::CYN, colors::RESET);

		SDL_ReleaseWindowFromGPUDevice(ctx.gpu.get(), ctx.window.get());

		ctx = {};
		SDL_Quit();
	}

	struct shader_desc
	{
		io::byte_array shader_binary;
		SDL_GPUShaderStage stage;
		uint32_t sampler_count         = 0;
		uint32_t uniform_buffer_count  = 0;
		uint32_t storage_buffer_count  = 0;
		uint32_t storage_texture_count = 0;
	};

	auto make_gpu_shader(SDL_GPUDevice *gpu, const shader_desc &desc) -> gpu_shader_ptr
	{
		auto backend_formats = SDL_GetGPUShaderFormats(gpu);
		assert(backend_formats & SDL_GPU_SHADERFORMAT_DXIL && "Backend shader format does not match DXIL");

		auto shader_format = SDL_GPU_SHADERFORMAT_DXIL;

		auto shader_info = SDL_GPUShaderCreateInfo{
			.code_size            = desc.shader_binary.size(),
			.code                 = reinterpret_cast<const uint8_t *>(desc.shader_binary.data()),
			.entrypoint           = "main",
			.format               = shader_format,
			.stage                = desc.stage,
			.num_samplers         = desc.sampler_count,
			.num_storage_textures = desc.storage_texture_count,
			.num_storage_buffers  = desc.storage_buffer_count,
			.num_uniform_buffers  = desc.uniform_buffer_count,
		};

		auto shader = SDL_CreateGPUShader(gpu, &shader_info);
		assert(shader != nullptr && "Failed to create shader.");

		return { shader, { gpu } };
	}

	struct pipeline_desc
	{
		shader_desc vertex;
		shader_desc fragment;

		std::span<const SDL_GPUVertexAttribute> vertex_attributes;
		std::span<const SDL_GPUVertexBufferDescription> vertex_buffer_descriptions;

		bool depth_test;
	};

	auto make_gfx_pipeline(const context &ctx, const pipeline_desc &desc) -> gfx_pipeline_ptr
	{
		auto gpu = ctx.gpu.get();
		auto wnd = ctx.window.get();

		std::println("{}Create pipeline.{}", colors::CYN, colors::RESET);

		auto vs_shdr = make_gpu_shader(gpu, desc.vertex);
		auto fs_shdr = make_gpu_shader(gpu, desc.fragment);

		auto vertex_input_state = SDL_GPUVertexInputState{
			.vertex_buffer_descriptions = desc.vertex_buffer_descriptions.data(),
			.num_vertex_buffers         = static_cast<uint32_t>(desc.vertex_buffer_descriptions.size()),
			.vertex_attributes          = desc.vertex_attributes.data(),
			.num_vertex_attributes      = static_cast<uint32_t>(desc.vertex_attributes.size()),
		};

		auto rasterizer_state = SDL_GPURasterizerState{
			.fill_mode  = SDL_GPU_FILLMODE_FILL,
			.cull_mode  = SDL_GPU_CULLMODE_BACK,
			.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
		};

		auto depth_stencil_state = SDL_GPUDepthStencilState{};
		if (desc.depth_test)
		{
			depth_stencil_state = SDL_GPUDepthStencilState{
				.compare_op          = SDL_GPU_COMPAREOP_LESS,
				.write_mask          = std::numeric_limits<uint8_t>::max(),
				.enable_depth_test   = true,
				.enable_depth_write  = true,
				.enable_stencil_test = false,
			};
		}

		auto color_targets = std::array{
			SDL_GPUColorTargetDescription{
			  .format = SDL_GetGPUSwapchainTextureFormat(gpu, wnd),
			},
		};

		auto target_info = SDL_GPUGraphicsPipelineTargetInfo{
			.color_target_descriptions = color_targets.data(),
			.num_color_targets         = static_cast<uint32_t>(color_targets.size()),
			.depth_stencil_format      = DEPTH_FORMAT,
			.has_depth_stencil_target  = desc.depth_test,
		};

		auto pipeline_info = SDL_GPUGraphicsPipelineCreateInfo{
			.vertex_shader       = vs_shdr.get(),
			.fragment_shader     = fs_shdr.get(),
			.vertex_input_state  = vertex_input_state,
			.primitive_type      = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
			.rasterizer_state    = rasterizer_state,
			.depth_stencil_state = depth_stencil_state,
			.target_info         = target_info,
		};
		auto pl = SDL_CreateGPUGraphicsPipeline(gpu, &pipeline_info);
		assert(pl != nullptr && "Failed to create graphics pipeline");

		return { pl, { gpu } };
	}

	auto make_buffer(SDL_GPUDevice *gpu, SDL_GPUBufferUsageFlags usage, uint32_t size, std::string_view name = ""sv) -> gpu_buffer_ptr
	{
		std::println("{}Create gpu buffer. {}/{}{}", colors::CYN, usage, size, colors::RESET);

		auto buffer_info = SDL_GPUBufferCreateInfo{
			.usage = usage,
			.size  = size,
		};

		auto buffer = SDL_CreateGPUBuffer(gpu, &buffer_info);
		assert(buffer != nullptr && "Failed to create gpu buffer");

		if (name.size() > 0)
		{
			SDL_SetGPUBufferName(gpu, buffer, name.data());
		}

		return { buffer, { gpu } };
	}

	struct texture_desc
	{
		SDL_GPUTextureUsageFlags usage;
		SDL_GPUTextureFormat format;
		uint32_t width;
		uint32_t height;
		uint32_t depth;
		uint32_t mip_levels;
	};

	auto make_texture(SDL_GPUDevice *gpu, const texture_desc &desc, std::string_view name = ""sv) -> gpu_texture_ptr
	{
		std::println("{}Create gpu texture. {}x{}x{}{}", colors::CYN, desc.width, desc.height, desc.depth, colors::RESET);

		auto texture_info = SDL_GPUTextureCreateInfo{
			.type                 = SDL_GPU_TEXTURETYPE_2D,
			.format               = desc.format,
			.usage                = desc.usage,
			.width                = desc.width,
			.height               = desc.height,
			.layer_count_or_depth = desc.depth,
			.num_levels           = desc.mip_levels,
		};

		auto texture = SDL_CreateGPUTexture(gpu, &texture_info);
		assert(texture != nullptr && "Failed to create gpu texture.");

		if (name.size() > 0)
		{
			SDL_SetGPUTextureName(gpu, texture, name.data());
		}

		return { texture, { gpu } };
	}

	enum class sampler_type
	{
		point_clamp,
		point_wrap,
		linear_clamp,
		linear_wrap,
		anisotropic_clamp,
		anisotropic_wrap,
	};

	auto to_string(sampler_type type) -> std::string_view
	{
		constexpr static auto type_names = std::array{
			"point_clamp"sv,
			"point_wrap"sv,
			"linear_clamp"sv,
			"linear_wrap"sv,
			"anisotropic_clamp"sv,
			"anisotropic_wrap"sv,
		};

		return type_names.at(static_cast<uint8_t>(type));
	}

	auto make_sampler(SDL_GPUDevice *gpu, sampler_type type) -> gpu_sampler_ptr
	{
		std::println("{}Create gpu sampler. {}{}", colors::CYN, to_string(type), colors::RESET);

		auto sampler_info = [&]() -> SDL_GPUSamplerCreateInfo {
			switch (type)
			{
			case sampler_type::point_clamp:
				return {
					.min_filter        = SDL_GPU_FILTER_NEAREST,
					.mag_filter        = SDL_GPU_FILTER_NEAREST,
					.mipmap_mode       = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
					.address_mode_u    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.address_mode_v    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.address_mode_w    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.max_anisotropy    = 0,
					.enable_anisotropy = false,
				};
			case sampler_type::point_wrap:
				return {
					.min_filter        = SDL_GPU_FILTER_NEAREST,
					.mag_filter        = SDL_GPU_FILTER_NEAREST,
					.mipmap_mode       = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
					.address_mode_u    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.address_mode_v    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.address_mode_w    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.max_anisotropy    = 0,
					.enable_anisotropy = false,
				};
			case sampler_type::linear_clamp:
				return {
					.min_filter        = SDL_GPU_FILTER_LINEAR,
					.mag_filter        = SDL_GPU_FILTER_LINEAR,
					.mipmap_mode       = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
					.address_mode_u    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.address_mode_v    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.address_mode_w    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.max_anisotropy    = 0,
					.enable_anisotropy = false,
				};
			case sampler_type::linear_wrap:
				return {
					.min_filter        = SDL_GPU_FILTER_LINEAR,
					.mag_filter        = SDL_GPU_FILTER_LINEAR,
					.mipmap_mode       = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
					.address_mode_u    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.address_mode_v    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.address_mode_w    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.max_anisotropy    = 0,
					.enable_anisotropy = false,
				};
			case sampler_type::anisotropic_clamp:
				return {
					.min_filter        = SDL_GPU_FILTER_LINEAR,
					.mag_filter        = SDL_GPU_FILTER_LINEAR,
					.mipmap_mode       = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
					.address_mode_u    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.address_mode_v    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.address_mode_w    = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
					.max_anisotropy    = MAX_ANISOTROPY,
					.enable_anisotropy = true,
				};
			case sampler_type::anisotropic_wrap:
				return {
					.min_filter        = SDL_GPU_FILTER_LINEAR,
					.mag_filter        = SDL_GPU_FILTER_LINEAR,
					.mipmap_mode       = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
					.address_mode_u    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.address_mode_v    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.address_mode_w    = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
					.max_anisotropy    = MAX_ANISOTROPY,
					.enable_anisotropy = true,
				};
			}

			return {};
		}();

		auto sampler = SDL_CreateGPUSampler(gpu, &sampler_info);
		assert(sampler != nullptr && "Failed to create sampler.");

		return { sampler, { gpu } };
	}

	struct scene
	{
		SDL_FColor clear_color;

		gfx_pipeline_ptr pipeline;

		gpu_buffer_ptr vertex_buffer;
		gpu_buffer_ptr index_buffer;
		gpu_buffer_ptr instance_buffer;
		uint32_t vertex_count;
		uint32_t index_count;
		uint32_t instance_count;

		gpu_texture_ptr depth_texture;
		gpu_texture_ptr uv_texture;
		gpu_sampler_ptr uv_sampler;

		io::byte_span view_projection;
	};

	void upload_to_gpu(SDL_GPUDevice *gpu,
	                   const io::byte_span vertices,
	                   const io::byte_span indices,
	                   const io::byte_span instances,
	                   const io::image_data &texture,
	                   scene &scn)
	{
		std::println("{}Upload to gpu memory.{}", colors::CYN, colors::RESET);

		auto tb_size = static_cast<uint32_t>(vertices.size() + indices.size() + instances.size() + texture.data.size());

		auto transfer_info = SDL_GPUTransferBufferCreateInfo{
			.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
			.size  = tb_size,
		};
		auto transfer_buffer = SDL_CreateGPUTransferBuffer(gpu, &transfer_info);
		assert(transfer_buffer != nullptr && "Failed to create gpu transfer buffer.");

		auto data = SDL_MapGPUTransferBuffer(gpu, transfer_buffer, false);

		// vertices
		std::memcpy(data, vertices.data(), vertices.size());
		// indicies
		data = io::offset_ptr(data, vertices.size());
		std::memcpy(data, indices.data(), indices.size());
		// instances
		data = io::offset_ptr(data, indices.size());
		std::memcpy(data, instances.data(), instances.size());
		// texture
		data = io::offset_ptr(data, instances.size());
		std::memcpy(data, texture.data.data(), texture.data.size());

		SDL_UnmapGPUTransferBuffer(gpu, transfer_buffer);

		std::println("{}Copy from gpu memory to gpu resources{}", colors::CYN, colors::RESET);
		auto copy_cmd  = SDL_AcquireGPUCommandBuffer(gpu);
		auto copy_pass = SDL_BeginGPUCopyPass(copy_cmd);

		auto offset = 0u;
		auto src_b  = SDL_GPUTransferBufferLocation{
			 .transfer_buffer = transfer_buffer,
			 .offset          = offset,
		};

		// vertices
		{
			auto dst = SDL_GPUBufferRegion{
				.buffer = scn.vertex_buffer.get(),
				.offset = 0,
				.size   = static_cast<uint32_t>(vertices.size()),
			};
			SDL_UploadToGPUBuffer(copy_pass, &src_b, &dst, false);
			offset += dst.size;
		}
		// indices
		{
			src_b.offset = offset;

			auto dst = SDL_GPUBufferRegion{
				.buffer = scn.index_buffer.get(),
				.offset = 0,
				.size   = static_cast<uint32_t>(indices.size()),
			};
			SDL_UploadToGPUBuffer(copy_pass, &src_b, &dst, false);
			offset += dst.size;
		}
		// instances
		{
			src_b.offset = offset;

			auto dst = SDL_GPUBufferRegion{
				.buffer = scn.instance_buffer.get(),
				.offset = 0,
				.size   = static_cast<uint32_t>(instances.size()),
			};
			SDL_UploadToGPUBuffer(copy_pass, &src_b, &dst, false);
			offset += dst.size;
		}
		// texture
		{
			// Copy data for each layer+mipmap in the array
			for (auto &&sub_image : texture.sub_images)
			{
				auto src_t = SDL_GPUTextureTransferInfo{
					.transfer_buffer = transfer_buffer,
					.offset          = offset + static_cast<uint32_t>(sub_image.offset),
				};

				auto dst = SDL_GPUTextureRegion{
					.texture   = scn.uv_texture.get(),
					.mip_level = sub_image.mipmap_index,
					.layer     = sub_image.layer_index,
					.w         = sub_image.width,
					.h         = sub_image.height,
					.d         = 1,
				};

				SDL_UploadToGPUTexture(copy_pass, &src_t, &dst, false);
			}
		}

		SDL_EndGPUCopyPass(copy_pass);
		SDL_SubmitGPUCommandBuffer(copy_cmd);
		SDL_ReleaseGPUTransferBuffer(gpu, transfer_buffer);
	}

	auto init_scene(const context &ctx,
	                const pipeline_desc &pipeline,
	                const io::byte_span vertices, uint32_t vertex_count,
	                const io::byte_span indices, uint32_t index_count,
	                const io::byte_span instances, uint32_t instance_count,
	                const io::image_data &texture) -> scene
	{
		auto gpu = ctx.gpu.get();
		auto w = 0, h = 0;
		SDL_GetWindowSizeInPixels(ctx.window.get(), &w, &h);

		auto scn = scene{
			.vertex_count   = vertex_count,
			.index_count    = index_count,
			.instance_count = instance_count,
		};

		scn.pipeline        = make_gfx_pipeline(ctx, pipeline);
		scn.vertex_buffer   = make_buffer(gpu, SDL_GPU_BUFFERUSAGE_VERTEX, static_cast<uint32_t>(vertices.size()), "Vertex Buffer"sv);
		scn.index_buffer    = make_buffer(gpu, SDL_GPU_BUFFERUSAGE_INDEX, static_cast<uint32_t>(indices.size()), "Index Buffer"sv);
		scn.instance_buffer = make_buffer(gpu, SDL_GPU_BUFFERUSAGE_VERTEX, static_cast<uint32_t>(instances.size()), "Instance Buffer"sv);

		auto td = texture_desc{
			.usage      = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
			.format     = DEPTH_FORMAT,
			.width      = static_cast<uint32_t>(w),
			.height     = static_cast<uint32_t>(h),
			.depth      = 1,
			.mip_levels = 1,
		};
		scn.depth_texture = make_texture(gpu, td, "Depth Texture"sv);

		auto td2 = texture_desc{
			.usage      = SDL_GPU_TEXTUREUSAGE_SAMPLER,
			.format     = texture.header.format,
			.width      = texture.header.width,
			.height     = texture.header.height,
			.depth      = texture.header.layer_count,
			.mip_levels = texture.header.mipmap_count,
		};
		scn.uv_texture = make_texture(gpu, td2, "UV texture"sv);
		scn.uv_sampler = make_sampler(gpu, sampler_type::anisotropic_clamp);

		upload_to_gpu(gpu, vertices, indices, instances, texture, scn);

		return scn;
	}

	void destroy_scene(scene &scn)
	{
		std::println("{}Destroy Scene.{}", colors::CYN, colors::RESET);

		scn = {};
	}

	// Get Swapchain Image/Texture, wait if none is available
	// Does not use smart pointer as lifetime of swapchain texture is managed by SDL
	auto get_swapchain_texture(SDL_Window *wnd, SDL_GPUCommandBuffer *cmd_buf) -> SDL_GPUTexture *
	{
		auto sc_tex = (SDL_GPUTexture *)nullptr;

		auto res = SDL_WaitAndAcquireGPUSwapchainTexture(cmd_buf, wnd, &sc_tex, NULL, NULL);
		assert(res == true && "Wait and acquire GPU swapchain texture failed.");
		assert(sc_tex != nullptr && "Swapchain texture is null. Is window minimized?");

		return sc_tex;
	}

	void draw(const context &ctx, const scene &scn, const io::byte_span view_proj)
	{
		auto gpu = ctx.gpu.get();
		auto wnd = ctx.window.get();

		auto cmd_buf = SDL_AcquireGPUCommandBuffer(gpu);
		assert(cmd_buf != nullptr && "Failed to acquire command buffer");

		// Push Uniform buffer
		SDL_PushGPUVertexUniformData(cmd_buf, 0, view_proj.data(), static_cast<uint32_t>(view_proj.size()));

		// Swapchain image
		auto sc_img = get_swapchain_texture(wnd, cmd_buf);

		auto color_target = SDL_GPUColorTargetInfo{
			.texture     = sc_img,
			.clear_color = scn.clear_color,
			.load_op     = SDL_GPU_LOADOP_CLEAR,
			.store_op    = SDL_GPU_STOREOP_STORE,
		};

		auto depth_target = SDL_GPUDepthStencilTargetInfo{
			.texture          = scn.depth_texture.get(),
			.clear_depth      = 1.0f,
			.load_op          = SDL_GPU_LOADOP_CLEAR,
			.store_op         = SDL_GPU_STOREOP_STORE,
			.stencil_load_op  = SDL_GPU_LOADOP_CLEAR,
			.stencil_store_op = SDL_GPU_STOREOP_STORE,
			.cycle            = true,
			.clear_stencil    = 0,
		};

		auto render_pass = SDL_BeginGPURenderPass(cmd_buf, &color_target, 1, nullptr /*&depth_target*/);
		{
			// Vertex and Instance buffer
			auto vertex_bindings = std::array{
				SDL_GPUBufferBinding{
				  .buffer = scn.vertex_buffer.get(),
				  .offset = 0,
				},
				SDL_GPUBufferBinding{
				  .buffer = scn.instance_buffer.get(),
				  .offset = 0,
				},
			};
			SDL_BindGPUVertexBuffers(render_pass, 0, vertex_bindings.data(), static_cast<uint32_t>(vertex_bindings.size()));

			// Index Buffer
			auto index_binding = SDL_GPUBufferBinding{
				.buffer = scn.index_buffer.get(),
				.offset = 0,
			};
			SDL_BindGPUIndexBuffer(render_pass, &index_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

			// UV Texture and Sampler
			auto sampler_binding = SDL_GPUTextureSamplerBinding{
				.texture = scn.uv_texture.get(),
				.sampler = scn.uv_sampler.get(),
			};
			SDL_BindGPUFragmentSamplers(render_pass, 0, &sampler_binding, 1);

			// Graphics Pipeline
			SDL_BindGPUGraphicsPipeline(render_pass, scn.pipeline.get());

			// Draw Indexed
			SDL_DrawGPUIndexedPrimitives(render_pass, scn.index_count, scn.instance_count, 0, 0, 0);
		}
		SDL_EndGPURenderPass(render_pass);

		SDL_SubmitGPUCommandBuffer(cmd_buf);
	}
}

namespace app
{
	auto quit = false;

	void update(float &angle)
	{
		auto *key_states = SDL_GetKeyboardState(nullptr);

		if (key_states[SDL_SCANCODE_ESCAPE])
			quit = true;

		if (key_states[SDL_SCANCODE_A])
			angle -= 0.5f;
		if (key_states[SDL_SCANCODE_D])
			angle += 0.5f;

		if (angle >= 360.0f or angle <= -360.0f)
			angle = 0.0f;
	}

	struct vertex
	{
		glm::vec3 pos;
		glm::vec2 uv;
	};

	struct mesh
	{
		std::vector<vertex> vertices;
		std::vector<uint32_t> indices;
	};

	auto make_cube() -> mesh
	{
		auto x = 1.f, y = 1.f, z = 1.f;

		auto vertices = std::vector<vertex>{
			// +X face
			{ { +x, -y, -z }, { 0.f, 1.f } },
			{ { +x, -y, +z }, { 0.f, 0.f } },
			{ { +x, +y, +z }, { 1.f, 0.f } },
			{ { +x, +y, -z }, { 1.f, 1.f } },
			// -X face
			{ { -x, -y, -z }, { 0.f, 1.f } },
			{ { -x, +y, -z }, { 1.f, 1.f } },
			{ { -x, +y, +z }, { 1.f, 0.f } },
			{ { -x, -y, +z }, { 0.f, 0.f } },
			// +Y face
			{ { -x, +y, -z }, { 0.f, 1.f } },
			{ { +x, +y, -z }, { 1.f, 1.f } },
			{ { +x, +y, +z }, { 1.f, 0.f } },
			{ { -x, +y, +z }, { 0.f, 0.f } },
			// -Y face
			{ { -x, -y, -z }, { 0.f, 1.f } },
			{ { -x, -y, +z }, { 0.f, 0.f } },
			{ { +x, -y, +z }, { 1.f, 0.f } },
			{ { +x, -y, -z }, { 1.f, 1.f } },
			// +Z face
			{ { -x, -y, +z }, { 0.f, 1.f } },
			{ { -x, +y, +z }, { 0.f, 0.f } },
			{ { +x, +y, +z }, { 1.f, 0.f } },
			{ { +x, -y, +z }, { 1.f, 1.f } },
			// -Z face
			{ { -x, -y, -z }, { 0.f, 1.f } },
			{ { +x, -y, -z }, { 1.f, 1.f } },
			{ { +x, +y, -z }, { 1.f, 0.f } },
			{ { -x, +y, -z }, { 0.f, 0.f } },
		};

		auto indices = std::vector<uint32_t>{
			0, 1, 2, 2, 3, 0,       // +X face
			4, 5, 6, 6, 7, 4,       // -X face
			8, 9, 10, 10, 11, 8,    // +Y face
			12, 13, 14, 14, 15, 12, // -Y face
			16, 17, 18, 18, 19, 16, // +Z face
			20, 21, 22, 22, 23, 20, // -Z face
		};

		return {
			vertices,
			indices,
		};
	}

	struct instance_data
	{
		std::vector<glm::mat4> transforms;
	};

	auto make_cube_instances() -> instance_data
	{
		auto cube_1 = glm::translate(glm::mat4(1.0f), glm::vec3{ 0.f, 0.f, 0.f });

		return {
			{ cube_1 },
		};
	}

	using VA                         = SDL_GPUVertexAttribute;
	constexpr auto VERTEX_ATTRIBUTES = std::array{
		VA{
		  .location    = 0,
		  .buffer_slot = 0,
		  .format      = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
		  .offset      = 0,
		},
		VA{
		  .location    = 1,
		  .buffer_slot = 0,
		  .format      = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
		  .offset      = sizeof(glm::vec3),
		},
		VA{
		  .location    = 2,
		  .buffer_slot = 1,
		  .format      = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
		  .offset      = 0,
		},
		VA{
		  .location    = 3,
		  .buffer_slot = 1,
		  .format      = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
		  .offset      = sizeof(glm::vec4),
		},
		VA{
		  .location    = 4,
		  .buffer_slot = 1,
		  .format      = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
		  .offset      = sizeof(glm::vec4) * 2,
		},
		VA{
		  .location    = 5,
		  .buffer_slot = 1,
		  .format      = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
		  .offset      = sizeof(glm::vec4) * 3,
		},
	};

	using VBD                          = SDL_GPUVertexBufferDescription;
	constexpr auto VERTEX_BUFFER_DESCS = std::array{
		VBD{
		  .slot       = 0,
		  .pitch      = sizeof(vertex),
		  .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
		},
		VBD{
		  .slot               = 1,
		  .pitch              = sizeof(glm::mat4),
		  .input_rate         = SDL_GPU_VERTEXINPUTRATE_INSTANCE,
		  .instance_step_rate = 1,
		},
	};

	auto get_pipeline_desc() -> sdl3::pipeline_desc
	{
		auto vs_bin = io::read_file("shaders/mesh.vs_6_4.cso");
		auto fs_bin = io::read_file("shaders/mesh.ps_6_4.cso");

		return {
			.vertex = sdl3::shader_desc{
			  .shader_binary        = vs_bin,
			  .stage                = SDL_GPU_SHADERSTAGE_VERTEX,
			  .uniform_buffer_count = 1,
			},
			.fragment = sdl3::shader_desc{
			  .shader_binary = fs_bin,
			  .stage         = SDL_GPU_SHADERSTAGE_FRAGMENT,
			  .sampler_count = 1,
			},
			.vertex_attributes          = VERTEX_ATTRIBUTES,
			.vertex_buffer_descriptions = VERTEX_BUFFER_DESCS,
			.depth_test                 = true,
		};
	}

	auto load_texture() -> io::image_data
	{
		return io::read_image_file("data/uv_grid.dds");
	}

	auto get_projection(uint32_t width, uint32_t height, float angle) -> glm::mat4
	{
		auto fov          = glm::radians(90.0f);
		auto aspect_ratio = static_cast<float>(width) / height;

		auto x = std::cosf(angle);
		auto z = std::sinf(angle);

		x = x * 2.5f;
		z = z * 2.5f;

		auto projection = glm::perspective(fov, aspect_ratio, 0.f, 100.f);
		auto view       = glm::lookAt(glm::vec3(x, 1.5, z),
		                              glm::vec3(0.f, 0.f, 0.f),
		                              glm::vec3(0.f, 1.f, 0.f));

		return projection * view;
	}
}

auto main() -> int
{
	constexpr auto width  = 800u;
	constexpr auto height = 600u;

	auto angle = 0.f;

	auto view_proj      = app::get_projection(width, height, glm::radians(angle));
	auto texture        = app::load_texture();
	auto cube_mesh      = app::make_cube();
	auto cube_instances = app::make_cube_instances();
	auto pl_desc        = app::get_pipeline_desc();

	auto ctx = sdl3::init_context(width, height, "SDL3 GPU Depth Texture"sv);
	auto scn = sdl3::init_scene(
		ctx,
		pl_desc,
		io::as_byte_span(cube_mesh.vertices), static_cast<uint32_t>(cube_mesh.vertices.size()),
		io::as_byte_span(cube_mesh.indices), static_cast<uint32_t>(cube_mesh.indices.size()),
		io::as_byte_span(cube_instances.transforms), static_cast<uint32_t>(cube_instances.transforms.size()),
		texture);

	scn.clear_color = { 0.4f, 0.4f, 0.4f, 1.0f };

	auto e = SDL_Event{};
	while (not app::quit)
	{
		while (SDL_PollEvent(&e))
		{
			if (e.type == SDL_EVENT_QUIT)
			{
				app::quit = true;
			}
		}
		sdl3::draw(ctx, scn, io::as_byte_span(view_proj));
		app::update(angle);

		view_proj = app::get_projection(width, height, glm::radians(angle));
	}

	sdl3::destroy_scene(scn);

	sdl3::destroy_context(ctx);
	return 0;
}
