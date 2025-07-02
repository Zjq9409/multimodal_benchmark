# 大语言模型性能测试脚本
1. 在不同硬件平台上启动vllm serving 服务.
2. 执行测试脚本，替换后续四个变量值
   ```
   python vllm_online_benchmark.py model_name max_seq input_len output_len
   ```

# 多模态性能测试脚本
1. 不同硬件平台启动多模态vllm serving服务
2. 运行测试脚本
```
model_path=xxxxx   # 替换成模型服务启动时的名称
for bs in 1 2 3 4 5 6; do
python ./vlm_benchmark.py \
--image_path ./resized_image.jpg  \
--prompt "简要描述图中的内容"      \
--model ${model_path} \
--batch_size ${bs} \
--port 8000 \
--host 127.0.0.1
done
```
如果指定了served-model-name名称，测试脚本如下
```
python ./vlm_benchmark.py  \
--image_path ./resized_image.jpg  \
--prompt "简要描述图中的内容" \
--model /data/models/Qwen2-VL-7B-Instruct \
--served-model-name Qwen2-VL-7B-Instruct \
--batch_size ${bs} \
--port 8000 \
--host 127.0.0.1
```
