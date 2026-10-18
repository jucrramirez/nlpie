fn main() {
    // With `openblas-static`, openblas-src compiles and links OpenBLAS itself;
    // no manual linkage (or system BLAS) is needed.
    #[cfg(not(feature = "openblas-static"))]
    {
        #[cfg(target_os = "macos")]
        {
            // Apple Accelerate framework provides BLAS/LAPACK.
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        #[cfg(target_os = "linux")]
        {
            // System OpenBLAS (install: sudo apt-get install libopenblas-dev gfortran).
            // If unavailable, rebuild with `--features openblas-static` instead.
            #[cfg(target_arch = "x86_64")]
            {
                add_linux_blas_search_paths("/usr/lib/x86_64-linux-gnu");
            }
            #[cfg(target_arch = "aarch64")]
            {
                add_linux_blas_search_paths("/usr/lib/aarch64-linux-gnu");
            }

            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:rustc-link-lib=dylib=gfortran");
        }
        #[cfg(target_os = "windows")]
        {
            println!(
                "cargo:warning=No system BLAS linkage is configured on Windows; \
                 build with `--features openblas-static`"
            );
        }
    }
}

#[cfg(target_os = "linux")]
fn add_linux_blas_search_paths(architecture_directory: &str) {
    println!("cargo:rustc-link-search={architecture_directory}");

    // Debian and Ubuntu package OpenBLAS variants in subdirectories such as
    // `openblas-pthread`, so the top-level architecture directory alone is
    // insufficient for the linker to resolve `-lopenblas`.
    for variant in ["openblas-pthread", "openblas-openmp", "openblas-serial"] {
        let path = format!("{architecture_directory}/{variant}");
        if std::path::Path::new(&path).is_dir() {
            println!("cargo:rustc-link-search={path}");
        }
    }
}
