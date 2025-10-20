# Rank: 0, Graph 37

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:497 in forward, code: self.alt_stream is not None
            stream = torch.cuda.streams.Stream(stream_id = 35, device_index = 0, device_type = 1);  stream = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:497 in forward, code: self.alt_stream is not None
                stream = torch.cuda.streams.Stream(stream_id = 35, device_index = 0, device_type = 1);  stream = None
                return ()
                