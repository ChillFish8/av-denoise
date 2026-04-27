use burn_onnx::ModelGen;

fn main() {
    println!("cargo:rerun-if-changed=models/");

    ModelGen::new()
        .input("models/rt_ldr_small.onnx")
        .out_dir("./src/model/")
        .run_from_cli();

    ModelGen::new()
        .input("models/rt_ldr.onnx")
        .out_dir("./src/model/")
        .run_from_cli();
}

