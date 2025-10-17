import torch
props = torch.cuda.get_device_properties(0)

print("GPU:", props.name)
print( props.max_threads_per_block)
print( props.max_threads_per_multiprocessor)
print( props.multi_processor_count)

