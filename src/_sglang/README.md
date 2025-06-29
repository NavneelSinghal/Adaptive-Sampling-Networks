### Recommended Workflow 

**Note :** 

- Add the HF_TOKEN from your HuggingFace account to the env.  Use `huggingface-cli login`
- Make sure you have all the requirements installed. Run `pip install -r requirements.txt` in the project root. 

**Steps :** 

- Launch Server: Open a terminal and start the sglang server. The most important parameter is the `--tp-size`. To use Tensor Parallelism use something bigger than 1. It will use that many machines to maximize throughput.
```
bash run_sglang_server.sh
```
- Run Parallel Client: Open a second terminal and run the parallel client script with a high --max_workers value. Default --max_workers are 16. Try larger values to maximise throughput. Decrease to manage memeory usage. It all depends on available VRAM. Play with this to figure out what will work for your scenario(model size and available VRAM across machines)
```
# Example of running the parallel client with 32 concurrent workers
# In your second terminal (while the server is running)
python src/_sglang/generate_rollouts_parallel.py \
  --heuristics_config_path ./configs/data_generation/heuristics_gemma3.yaml \
  --source_data_path ./data/test_prompts.jsonl \
  --output_path ./data/test_outputs.jsonl
  --max_workers 32
```  

- To configure the server, make sure you use the correct model name from hugging face in the bash script. If your heuristics include any of `eta` `epsilon` or `typical` sampling techniques, make sure to use the `--enable-custom-logit-processor` tag so that we can use the sglang native implementations in the `sglang_custom_processors.py` file. 


 

