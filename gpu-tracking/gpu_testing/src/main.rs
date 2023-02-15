use wgpu;
use gpu_tracking::execute_gpu::path_to_iter;
use ndarray::Array;

fn main() {
    // let path = "testing/easy_test_data.tif";
    // let (provider, dims) = path_to_iter(&path, None).unwrap();
    // let data = provider.into_iter().flat_map(|vec| vec.unwrap().into_iter()).collect::<Vec<_>>();
    // let array = Array::from_shape_vec([data.len() / (dims[0] * dims[1]) as usize, dims[0] as usize, dims[1] as usize], data).unwrap();

    let instance = wgpu::Instance::new(
        wgpu::InstanceDescriptor{
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        }
    );

    for adapt in instance.enumerate_adapters(wgpu::Backends::all()){
        dbg!(adapt.get_info());
        println!("")
    }
    // dbg!(array.shape());
}
