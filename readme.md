- 首先启动vllm大模型服务端

- 运行测试脚本：
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
