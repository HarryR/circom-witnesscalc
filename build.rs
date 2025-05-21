use std::env;
use std::io::Result;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    if let Ok(target_os) = std::env::var("CARGO_CFG_TARGET_OS") {
        if target_os == "android" {
            let target = std::env::var("TARGET").expect("TARGET is not set");
            let cc_target = format!("CC_{}", target);
            if let Ok(clang_path) = std::env::var(cc_target) {
                std::env::set_var("CC", clang_path.clone());
                std::env::set_var("CLANG_PATH", clang_path.clone());
            } else {
                if let Ok(_cc_path) = std::env::var("CC") {
                } else {
                    panic!("Please run with cargo-ndk or set CC environment variable");
                }

                if let Ok(_clang_path) = std::env::var("CLANG_PATH") {
                } else {
                    panic!("Please run with cargo-ndk or set CLANG_PATH environment variable");
                }
            }
        }
    }

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("include/graph_witness.h")
        .allowlist_type("gw_status_t")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Generate protobuf bindings
    let empty_array: &[&Path] = &[];
    println!("cargo:rerun-if-changed=protos/messages.proto");
    prost_build::compile_protos(&["protos/messages.proto"], empty_array)?;
    println!("cargo:rerun-if-changed=protos/vm.proto");
    prost_build::compile_protos(&["protos/vm.proto"], empty_array)?;

    // println!("cargo:rustc-link-arg=-Wl,-soname,libcircom_witnesscalc.so");

    let target = env::var("TARGET").unwrap();
    if target.contains("android") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libcircom_witnesscalc.so");
    }

    Ok(())
}
