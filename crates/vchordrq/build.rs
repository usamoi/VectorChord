fn main() {
    println!("cargo::rerun-if-changed=cuda");
    cc::Build::new()
        .file("cuda/vchordrq_assign.cu")
        .cuda(true)
        .ccbin(false)
        .cudart("static")
        .cpp(true)
        .std("c++17")
        .flags(["-gencode", "arch=compute_80,code=sm_80"])
        .flags(["-gencode", "arch=compute_86,code=sm_86"])
        .flags(["-gencode", "arch=compute_87,code=sm_87"])
        .flags(["-gencode", "arch=compute_89,code=sm_89"])
        .flags(["-gencode", "arch=compute_90,code=sm_90"])
        .flags(["-gencode", "arch=compute_100,code=sm_100"])
        .flags(["-gencode", "arch=compute_101,code=sm_101"])
        // .flags(["-gencode", "arch=compute_103,code=sm_103"])
        .flags(["-gencode", "arch=compute_120,code=sm_120"])
        // .flags(["-gencode", "arch=compute_121,code=sm_121"])
        .flag("-t0")
        .compile("vchordrq_assign");
}
