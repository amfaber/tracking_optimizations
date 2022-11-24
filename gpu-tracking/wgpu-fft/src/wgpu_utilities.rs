use std::rc::Rc;

pub struct FullComputePass{
    pub bindgroup: wgpu::BindGroup,
    pub wg_n: [u32; 3],
    pub pipeline: Rc<wgpu::ComputePipeline>,
}


impl FullComputePass{
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &self.bindgroup, &[]);
        cpass.set_pipeline(&self.pipeline);
        if push_constants.len() > 0{
            cpass.set_push_constants(0, push_constants);
        }
        cpass.dispatch_workgroups(self.wg_n[0], self.wg_n[1], self.wg_n[2]);
    }
}


pub fn infer_compute_bindgroup_layout(device: &wgpu::Device, source: &str) -> wgpu::BindGroupLayout{
    let re = regex::Regex::new(r"@binding\((?P<idx>\d)\)\s*var<(?P<type>.*?)>").unwrap();

    let mut entries = Vec::new();
    for capture in re.captures_iter(source){
        let idx: u32 = capture.name("idx").expect("Regex failed parse at binding idx").as_str().parse().unwrap();
        let ty = capture.name("type").expect("Regex failed parse at binding type").as_str();
        let ty = match ty{
            "uniform" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            "storage, read" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            "storage, read_write" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            _ => panic!("Unrecognized binding type: {}", ty)
        };
        entries.push(
            wgpu::BindGroupLayoutEntry{
                binding: idx,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        );
    }
    let bindgrouplayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &entries[..]
    });

    bindgrouplayout
    
}