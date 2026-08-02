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
            println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
            #[cfg(target_arch = "aarch64")]
            println!("cargo:rustc-link-search=/usr/lib/aarch64-linux-gnu");

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
