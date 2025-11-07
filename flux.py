from diffusers import FluxPipeline
import os
import time
import torch

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

models = {
    "FLUX.1-schnell": "black-forest-labs/FLUX.1-schnell",
}

prompts = [
  """
  A fearsome dinosaur Space Marine (Raptor) in heavily battle-damaged power armor, 
  wielding a massive bolter and a chainsword, locked in fierce combat against a horde of alien xenomorphs on the lunar surface. 
  The desolate moon landscape with craters and the giant Earth looming in the black sky. 
  Explosions and laser blasts illuminate the scene. 
  cinematic lighting, highly detailed, digital painting, in the style of Warhammer 40k art and Greg Rutkowski.
  """
]


def benchmark_models(name: str, model: str, prompts: list) -> None:

    load_start_time = time.perf_counter()

    pipe = FluxPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map="balanced",
    )
    load_end_time = time.perf_counter()
    print(f"Model loaded in {load_end_time - load_start_time:.2f} sec")

    pipe.enable_vae_slicing()

    for n, prompt in enumerate(prompts):

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            image = pipe(
                prompt,
                height=512,
                width=512,
                num_inference_steps=5,
                guidance_scale=1.0,
            ).images[0]

        torch.cuda.empty_cache() 
        times = []
        gpu_memory_peaks = []
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated()
        
        times.append(execution_time)
        gpu_memory_peaks.append(peak_memory)
        
        print(f"⏱️  Time: {execution_time:.2f} sec")
        print(f" GPU Memory Peak: {peak_memory / (1024**3):.2f} GB")

        image.save(f"{name}_{n}.png")


if __name__ == "__main__":
    for name, model in models.items():
        benchmark_models(name, model, prompts)
